import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import json
import random
import argparse
from tqdm import tqdm
from tools.data_utils import load_jsonl, set_seed

class DenoisingDataset(Dataset):
    def __init__(self, data_path=None, size=1000):
                   
        if data_path and os.path.exists(data_path):
            self.data = self._load_from_file(data_path)
            print(f"Loaded {len(self.data)} samples from {data_path}")
        else:
            print(f"Using synthetic data (size={size})")
            self.data = [
                {
                    "query": "Who is the director of Avatar?",
                    "positives": ["James Cameron directed Avatar."],
                    "negatives": ["Avatar is a blue alien movie."] * 99
                } for _ in range(size)
            ]

    def _load_from_file(self, data_path):
                            
        data = []
        if data_path.endswith('.jsonl'):
            items = load_jsonl(data_path)
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                items = json.load(f)
        
        for item in items:
            query = item.get('question', item.get('query', ''))
            if not query:
                continue
                
            positives = []
            if 'supporting_facts' in item and item['supporting_facts']:
                for fact in item['supporting_facts']:
                    if isinstance(fact, list) and len(fact) >= 2:
                        title, sent_idx = fact[0], fact[1]
                        if 'context' in item and isinstance(item['context'], list):
                            for ctx in item['context']:
                                if isinstance(ctx, list) and len(ctx) > 1 and ctx[0] == title:
                                    if isinstance(ctx[1], list) and sent_idx < len(ctx[1]):
                                        positives.append(str(ctx[1][sent_idx]))
                                    break
            
            if not positives and 'context' in item and isinstance(item['context'], list):
                for ctx in item['context'][:3]:
                    if isinstance(ctx, list) and len(ctx) > 1:
                        if isinstance(ctx[1], list):
                            ctx_text = " ".join(str(s) for s in ctx[1][:2])
                        else:
                            ctx_text = str(ctx[1])
                        if ctx_text and len(ctx_text.strip()) > 10:
                            positives.append(ctx_text)
            
            negatives = []
            if 'context' in item and isinstance(item['context'], list):
                for ctx in item['context']:
                    if isinstance(ctx, list) and len(ctx) > 1:
                        if isinstance(ctx[1], list):
                            ctx_text = " ".join(str(s) for s in ctx[1][:2])
                        else:
                            ctx_text = str(ctx[1])
                        if ctx_text and len(ctx_text.strip()) > 10 and ctx_text not in positives:
                            negatives.append(ctx_text)
            
            if positives and negatives:
                data.append({
                    "query": query,
                    "positives": positives,
                    "negatives": negatives[:50]
                })
        
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = str(item['query']) if item.get('query') else ""
        pos_doc = str(random.choice(item['positives'])) if item.get('positives') else "Dummy positive"
        neg_doc = str(random.choice(item['negatives'])) if item.get('negatives') else "Dummy negative"
        return query, pos_doc, neg_doc

def collate_fn_strings(batch):
                            
    queries, pos_docs, neg_docs = zip(*batch)
    return list(queries), list(pos_docs), list(neg_docs)

class RelevanceAgent(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

    def forward(self, q_emb, d_emb):
        features = torch.cat([q_emb, d_emb, torch.abs(q_emb - d_emb)], dim=1)
        return self.net(features)

def train_stage2(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")
    
    print("Loading Encoder (BGE-Large)...")
    local_model_path = os.path.join(_project_root, "LLM/BAAI/bge-large-en-v1.5")
    if os.path.exists(local_model_path):
        print(f"  Using local model: {local_model_path}")
        encoder = SentenceTransformer(local_model_path, device=device)
    else:
        print("  Using HuggingFace model: BAAI/bge-large-en-v1.5")
        encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
    encoder.eval()
    
    agent = RelevanceAgent().to(device)
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None
    optimizer = optim.AdamW(agent.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    dataset = DenoisingDataset(data_path=args.data_path, size=args.synthetic_size)
    num_workers = min(8, (os.cpu_count() or 1) // 2) if args.batch_size >= 128 else 0
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=collate_fn_strings,
        pin_memory=True if device == "cuda" else False,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"Start Training Relevance Agent...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Total samples: {len(dataset)}")
    
    agent.train()
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                queries, pos_docs, neg_docs = batch
                
                queries = [str(q).strip() for q in queries if q and str(q).strip()]
                pos_docs = [str(d).strip() for d in pos_docs if d and str(d).strip()]
                neg_docs = [str(d).strip() for d in neg_docs if d and str(d).strip()]
                
                if not queries or not pos_docs or not neg_docs:
                    continue
                
                min_len = min(len(queries), len(pos_docs), len(neg_docs))
                queries = queries[:min_len]
                pos_docs = pos_docs[:min_len]
                neg_docs = neg_docs[:min_len]
                
                with torch.no_grad():
                    encode_batch_size = min(512, len(queries))
                    q_embs = encoder.encode(queries, convert_to_tensor=True, show_progress_bar=False, batch_size=encode_batch_size).to(device)
                    pos_embs = encoder.encode(pos_docs, convert_to_tensor=True, show_progress_bar=False, batch_size=encode_batch_size).to(device)
                    neg_embs = encoder.encode(neg_docs, convert_to_tensor=True, show_progress_bar=False, batch_size=encode_batch_size).to(device)
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                print(f"  Batch type: {type(batch)}")
                print(f"  Queries type: {type(queries) if 'queries' in locals() else 'N/A'}")
                if 'queries' in locals():
                    print(f"  Queries sample: {queries[:2] if len(queries) > 0 else 'Empty'}")
                raise
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    pos_scores = agent(q_embs, pos_embs).squeeze()
                    neg_scores = agent(q_embs, neg_embs).squeeze()
                
                loss_pos = criterion(pos_scores, torch.ones_like(pos_scores))
                loss_neg = criterion(neg_scores, torch.zeros_like(neg_scores))
                loss = loss_pos + loss_neg
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pos_scores = agent(q_embs, pos_embs).squeeze()
                loss_pos = criterion(pos_scores, torch.ones_like(pos_scores))
                
                neg_scores = agent(q_embs, neg_embs).squeeze()
                loss_neg = criterion(neg_scores, torch.zeros_like(neg_scores))
                
                loss = loss_pos + loss_neg
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "relevance_agent.pth")
    torch.save(agent.state_dict(), output_path)
    print(f"Stage 2 Training Completed. Model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 2: Diffusion Denoising Agent")
    parser.add_argument("--data_path", type=str, default=None, help="Path to training data (JSON/JSONL)")
    parser.add_argument("--output_dir", type=str, default="outputs/stage2", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--synthetic_size", type=int, default=1000, help="Size of synthetic data if no data_path")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (even if CUDA available)")
    
    args = parser.parse_args()
    train_stage2(args)

