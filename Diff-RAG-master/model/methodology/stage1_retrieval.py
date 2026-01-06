import networkx as nx
import spacy
import torch
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import os

nlp = spacy.load("en_core_web_sm")


class QueryPreprocessor:
   
    def __init__(self):
        self.nlp = nlp
    
    def preprocess(self, query: str, conversation_history: Optional[List[str]] = None) -> str:
       
        if conversation_history:
            context = " ".join(conversation_history[-3:])
            processed_text = f"{context} {query}"
        else:
            processed_text = query
        
        doc = self.nlp(processed_text)
        key_phrases = [chunk.text for chunk in doc.noun_chunks]
        entities = [ent.text for ent in doc.ents]
        
        if entities:
            rewritten_query = f"{query} {' '.join(entities[:3])}"
        else:
            rewritten_query = query
        
        return processed_text, rewritten_query


class DenseRetriever:
   
    def __init__(self, encoder_path: Optional[str] = None, device: str = "cuda"):
       
        self.device = device
        
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        local_model_path = os.path.join(_project_root, "LLM/BAAI/bge-large-en-v1.5")
        
        if encoder_path:
            self.encoder = SentenceTransformer(encoder_path, device=device)
        elif os.path.exists(local_model_path):
            print(f"  Using local encoder: {local_model_path}")
            self.encoder = SentenceTransformer(local_model_path, device=device)
        else:
            print("  Using HuggingFace encoder: BAAI/bge-large-en-v1.5")
            self.encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
        
        self.encoder.eval()
        self.index = None
        self.passage_embeddings = None
        self.passages = []
    
    def build_index(self, passages: List[str], index_type: str = "flat"):
        
        print(f"Building Faiss index for {len(passages)} passages...")
        
        self.passages = passages
        self.passage_embeddings = self.encoder.encode(
            passages, 
            convert_to_numpy=True, 
            show_progress_bar=True,
            batch_size=128
        )
        
        dimension = self.passage_embeddings.shape[1]
        
        if index_type == "flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "ivf":
            nlist = 100
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.index.train(self.passage_embeddings)
        
        self.index.add(self.passage_embeddings.astype('float32'))
        print(f"Index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = 100) -> Tuple[List[str], torch.Tensor]:
        
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        query_emb = self.encoder.encode(query, convert_to_numpy=True)
        query_emb = query_emb.reshape(1, -1).astype('float32')
        
        distances, indices = self.index.search(query_emb, top_k)
        
        retrieved_passages = [self.passages[i] for i in indices[0]]
        retrieved_embeddings = self.passage_embeddings[indices[0]]
        
        return retrieved_passages, torch.tensor(retrieved_embeddings, device=self.device)


class GraphRetriever:
   
    def __init__(self):
        self.graph_db = None
    
    def build_graph_db(self, passages: List[str], graph: Optional[nx.Graph] = None):
       
        if graph is not None:
            self.graph_db = graph
        else:
            self.graph_db = nx.Graph()
            for passage in passages:
                doc = nlp(passage)
                entities = [e.text for e in doc.ents]
                for i, ent1 in enumerate(entities):
                    for ent2 in entities[i+1:]:
                        if not self.graph_db.has_edge(ent1, ent2):
                            self.graph_db.add_edge(ent1, ent2, weight=1.0)
                        else:
                            self.graph_db[ent1][ent2]['weight'] += 1.0
    
    def search_kg_subgraphs(self, query: str, top_k: int = 10) -> List[nx.Graph]:
       
        if self.graph_db is None:
            return []
        
        doc = nlp(query)
        query_entities = [e.text for e in doc.ents]
        
        subgraphs = []
        for entity in query_entities[:top_k]:
            if entity in self.graph_db:
                neighbors = list(self.graph_db.neighbors(entity))
                if neighbors:
                    subgraph = self.graph_db.subgraph([entity] + neighbors[:5])
                    subgraphs.append(subgraph)
        
        return subgraphs


class DynamicGraphBuilder:
   
    def __init__(self):
        self.graph = nx.Graph()
        
    def build(self, passages: List[str]):
        self.graph.clear()
        doc_entities = []

        for idx, text in enumerate(passages):
            doc = nlp(text)
            ents = list(set([e.text for e in doc.ents]))
            doc_entities.append(ents)
            
            for ent in ents:
                self.graph.add_node(ent)
        
        for ents in doc_entities:
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    self.graph.add_edge(ents[i], ents[j])
        
        return self.graph


class HybridRetriever:
   
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.query_preprocessor = QueryPreprocessor()
        self.dense_retriever = DenseRetriever(device=device)
        self.graph_retriever = GraphRetriever()
        self.graph_builder = DynamicGraphBuilder()
        self.nlp = nlp
        
    def search(self, query: str, corpus: List[str] = None, top_k: int = 100, 
               conversation_history: Optional[List[str]] = None) -> Tuple[List[str], torch.Tensor, nx.Graph]:
        processed_text, rewritten_query = self.query_preprocessor.preprocess(query, conversation_history)
        
        if corpus is not None:
            self.dense_retriever.build_index(corpus)
        
        dense_passages, dense_embs = self.dense_retriever.search(rewritten_query, top_k=top_k)
        
        self.graph_retriever.build_graph_db(dense_passages)
        kg_subgraphs = self.graph_retriever.search_kg_subgraphs(rewritten_query, top_k=10)
        
        candidates = dense_passages
        candidate_embs = dense_embs
        
        local_graph = self.graph_builder.build(candidates)
        
        for subgraph in kg_subgraphs:
            local_graph = nx.compose(local_graph, subgraph)
        
        return candidates, candidate_embs, local_graph
    
    @property
    def encoder(self):
        return self.dense_retriever.encoder