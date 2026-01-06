import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Optional
from sentence_transformers import SentenceTransformer


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


class DiffusionDenoisingKernel:
           
    def __init__(self, device='cuda', relevance_agent_path: str = None, encoder_path: str = None):
                   
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
        
        if not relevance_agent_path:
            raise ValueError(
                "relevance_agent_path is required. Please provide the path to the trained Relevance Agent model. "
                "Train the model using: python model/experiments/train_stage2.py"
            )
        
        if not os.path.exists(relevance_agent_path):
            raise FileNotFoundError(
                f"Relevance Agent model not found at: {relevance_agent_path}\n"
                f"Please train the model first using: python model/experiments/train_stage2.py"
            )
        
        print(f"Loading trained Relevance Agent from {relevance_agent_path}")
        # Note: don't name this attribute `relevance_agent`, to avoid shadowing methods.
        self.relevance_model = RelevanceAgent(input_dim=1024).to(device)
        self.relevance_model.load_state_dict(torch.load(relevance_agent_path, map_location=device))
        self.relevance_model.eval()
        
    def scheduler_agent(self, t, T, k_max: int = 10, k_min: int = 3):
                   
        progress = (T - t) / T
        k_t = int(k_max - (k_max - k_min) * progress)
        tau_t = 1.0 - 0.9 * progress 
        alpha_t = 0.5 + 0.4 * progress 
        return tau_t, k_t, alpha_t

    def compute_relevance_scores(self, query_emb: torch.Tensor, doc_embs: torch.Tensor) -> torch.Tensor:
        """Return sigmoid relevance scores for each document (shape: [M])."""
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        if query_emb.size(0) != 1:
            # This kernel currently assumes a single query embedding.
            query_emb = query_emb[:1]

        M = doc_embs.shape[0]
        q_expanded = query_emb.expand(M, -1)
        logits = self.relevance_model(q_expanded, doc_embs).squeeze(-1)  # [M]
        return torch.sigmoid(logits)

    def consistency_agent(self, active_indices, doc_embs):
                                   
        if len(active_indices) <= 1:
            return torch.ones(len(active_indices)).to(self.device)
            
        subset_embs = doc_embs[active_indices]
        sim_matrix = torch.matmul(subset_embs, subset_embs.T)
        centrality = sim_matrix.mean(dim=1)
        return torch.sigmoid(centrality)

    def run_diffusion(
        self,
        query_emb: torch.Tensor,
        doc_embs: torch.Tensor,
        T: int = 10,
        final_k_min: int = 3,
        final_k_max: int = 10,
        final_cumprob: float = 0.8,
        schedule_k_min: Optional[int] = None,
        schedule_k_max: Optional[int] = None,
    ) -> torch.Tensor:
                   
        M = doc_embs.shape[0]
        initial_scores = torch.matmul(query_emb, doc_embs.T).squeeze()
        z = F.softmax(initial_scores, dim=0)

        if schedule_k_min is None:
            schedule_k_min = final_k_min
        if schedule_k_max is None:
            schedule_k_max = final_k_max
        
        for t in range(T, 0, -1):
            tau, k_t, alpha = self.scheduler_agent(t, T, k_max=schedule_k_max, k_min=schedule_k_min)
            
            s = self.compute_relevance_scores(query_emb, doc_embs)
            z_hat = alpha * z + (1 - alpha) * s
            
            _, active_indices = torch.topk(z_hat, k_t)
            consistency_weights = self.consistency_agent(active_indices, doc_embs)
            
            z_update = z_hat.clone()
            z_update[active_indices] = z_hat[active_indices] * consistency_weights
            z = z_update
            
            z = z / z.sum()

        # Dynamically choose final_k in [final_k_min, final_k_max] based on cumulative mass of z.
        sorted_z, _ = torch.sort(z, descending=True)
        cum = torch.cumsum(sorted_z, dim=0)
        # smallest k where cumulative probability exceeds threshold
        if final_cumprob <= 0:
            k_final = final_k_min
        elif final_cumprob >= 1:
            k_final = min(M, final_k_max)
        else:
            k_final = int((cum >= final_cumprob).nonzero(as_tuple=False).min().item() + 1) if (cum >= final_cumprob).any() else M
        k_final = max(final_k_min, min(final_k_max, min(M, k_final)))

        _, golden_indices = torch.topk(z, k_final)
        return golden_indices