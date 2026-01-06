import torch
import torch.nn as nn
import os
from sentence_transformers import SentenceTransformer
from typing import List, Optional


class PathLinearizer:
    def linearize(self, passage_text, graph, nlp_obj):
        doc = nlp_obj(passage_text)
        entities = [e.text for e in doc.ents]
        
        neighbors = set()
        for ent in entities:
            if ent in graph:
                nbrs = list(graph.neighbors(ent))
                neighbors.update(nbrs)
        
        if not neighbors:
            return ""
            
        path_str = ", ".join(list(neighbors)[:5])
        return f" [SEP] Structurally associated with: {path_str}."


class KBEncoder:
    
    def __init__(self, encoder_path: Optional[str] = None, device: str = "cuda"):
    
        self.device = device
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        local_model_path = os.path.join(_project_root, "LLM/BAAI/bge-large-en-v1.5")
        
        if encoder_path:
            self.encoder = SentenceTransformer(encoder_path, device=device)
        elif os.path.exists(local_model_path):
            print(f"  Using local KB encoder: {local_model_path}")
            self.encoder = SentenceTransformer(local_model_path, device=device)
        else:
            print("  Using HuggingFace KB encoder: BAAI/bge-large-en-v1.5")
            self.encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
        
        self.encoder.eval()
    
    def encode(self, evidence_texts: List[str], convert_to_tensor: bool = True) -> torch.Tensor:
       
        evidence_embs = self.encoder.encode(
            evidence_texts,
            convert_to_tensor=convert_to_tensor,
            show_progress_bar=False,
            batch_size=32
        )
        if convert_to_tensor:
            evidence_embs = evidence_embs.to(self.device)
        return evidence_embs


class KnowledgeAdapter(nn.Module):
   
    def __init__(self, encoder_dim=1024, llm_dim=3584):
       
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, llm_dim),
            nn.SiLU(),
            nn.Linear(llm_dim, llm_dim)
        )
        
        self.norm = nn.LayerNorm(llm_dim)       
        self.v_z = nn.Sequential(
            nn.Linear(llm_dim, llm_dim),
            nn.Tanh(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, encoder_output):
        x = self.projector(encoder_output)
        x = self.norm(x)
        vectors = self.v_z(x)       
        return vectors


def inject_knowledge_to_retrieval_layer(
    adapter: KnowledgeAdapter,
    kb_encoder: KBEncoder,
    golden_evidence: List[str],
    llm_input_embeds: torch.Tensor,
    retrieval_layer_idx: int = None
) -> torch.Tensor:
    evidence_embs = kb_encoder.encode(golden_evidence, convert_to_tensor=True)
    batch_size = llm_input_embeds.shape[0]
    evidence_embs = evidence_embs.unsqueeze(0).expand(batch_size, -1, -1)   
    virtual_tokens = adapter(evidence_embs)   
    injected_embeds = torch.cat([virtual_tokens, llm_input_embeds], dim=1)   
    return injected_embeds


def inject_knowledge(adapter, encoder_output, llm_input_embeds):
    virtual_tokens = adapter(encoder_output)
    combined_embeds = torch.cat([virtual_tokens, llm_input_embeds], dim=1)
    return combined_embeds