import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
import numpy as np


class KVCache:
           
    def __init__(self):
        self.cache = {}
    
    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
                   
        if layer_idx not in self.cache:
            self.cache[layer_idx] = (key, value)
        else:
            old_key, old_value = self.cache[layer_idx]
            new_key = torch.cat([old_key, key], dim=1)
            new_value = torch.cat([old_value, value], dim=1)
            self.cache[layer_idx] = (new_key, new_value)
    
    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
                             
        return self.cache.get(layer_idx, None)
    
    def clear(self):
                        
        self.cache.clear()


class SupervisedAttentionMechanism(nn.Module):
           
    def __init__(self, hidden_dim: int, num_heads: int = 8):
                   
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        golden_kb_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = query.shape
        
        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)
        
        if golden_kb_kv is not None:
            kb_k, kb_v = golden_kb_kv
            K = torch.cat([kb_k, K], dim=1)
            V = torch.cat([kb_v, V], dim=1)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)        
        output = torch.matmul(attention_weights, V)       
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)       
        return output, attention_weights


def focal_loss(
    attention_map: torch.Tensor,
    target_mask: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0
) -> torch.Tensor:
    ce_loss = F.binary_cross_entropy(attention_map, target_mask, reduction='none')
    
    p_t = attention_map * target_mask + (1 - attention_map) * (1 - target_mask)
    focal_weight = (1 - p_t) ** gamma    
    alpha_t = alpha * target_mask + (1 - alpha) * (1 - target_mask)   
    focal_loss = alpha_t * focal_weight * ce_loss
    return focal_loss.mean()


def create_supervised_attention_mask(
    batch_size: int,
    num_evidence: int,
    query_len: int,
    answer_len: int,
    device: str = 'cuda'
) -> torch.Tensor:
    total_len = num_evidence + query_len + answer_len
    
    mask = torch.zeros((batch_size, total_len, total_len), device=device)
    
    idx_ev_end = num_evidence
    idx_q_end = num_evidence + query_len
    
    mask[:, :idx_ev_end, :idx_ev_end] = 1
    
    mask[:, idx_ev_end:idx_q_end, :idx_q_end] = 1
    
    mask[:, idx_q_end:, :idx_ev_end] = 1
    mask[:, idx_q_end:, idx_ev_end:idx_q_end] = 1
    causal_part = torch.tril(torch.ones((answer_len, answer_len), device=device))
    mask[:, idx_q_end:, idx_q_end:] = causal_part.unsqueeze(0).expand(batch_size, -1, -1)
    
    return mask


def nucleus_sampling(
    logits: torch.Tensor,
    top_p: float = 0.9,
    temperature: float = 1.0
) -> torch.Tensor:
    logits = logits / temperature
    
    probs = F.softmax(logits, dim=-1)
    
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_indices_to_remove = cumsum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    probs[indices_to_remove] = 0
    
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    sampled_indices = torch.multinomial(probs, num_samples=1)
    return sampled_indices.squeeze(-1)


def beam_search(
    model,
    input_ids: torch.Tensor,
    beam_width: int = 5,
    max_length: int = 100,
    kv_cache: Optional[KVCache] = None
) -> List[torch.Tensor]:
    batch_size = input_ids.shape[0]
    device = input_ids.device
    
    beams = [(input_ids, 0.0)]
    
    for step in range(max_length):
        new_beams = []
        
        for sequence, score in beams:
            with torch.no_grad():
                outputs = model(sequence)
                next_token_logits = outputs.logits[:, -1, :]
            
            top_k_logits, top_k_indices = torch.topk(next_token_logits, beam_width, dim=-1)
            
            for k in range(beam_width):
                token_id = top_k_indices[0, k].unsqueeze(0).unsqueeze(0)
                token_score = top_k_logits[0, k].item()
                new_sequence = torch.cat([sequence, token_id], dim=1)
                new_score = score + token_score
                new_beams.append((new_sequence, new_score))
        
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        
        if any(seq[0, -1].item() == model.config.eos_token_id for seq, _ in beams):
            break
    
    return [seq for seq, _ in beams]


def generate_with_supervised_attention(
    model,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    golden_kb_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    kv_cache: Optional[KVCache] = None,
    max_new_tokens: int = 50,
    sampling_method: str = "nucleus",
    top_p: float = 0.9,
    temperature: float = 1.0
) -> torch.Tensor:
    model.eval()
    device = inputs_embeds.device
    batch_size = inputs_embeds.shape[0]
    
    generated_ids = []
    current_input = inputs_embeds
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model(inputs_embeds=current_input, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            
            if sampling_method == "nucleus":
                next_token = nucleus_sampling(logits, top_p=top_p, temperature=temperature)
            else:
                next_token = torch.argmax(logits, dim=-1)
            
            generated_ids.append(next_token)
            
            if next_token.item() == model.config.eos_token_id:
                break
            
            next_token_embed = model.get_input_embeddings()(next_token.unsqueeze(1))
            current_input = torch.cat([current_input, next_token_embed], dim=1)
            
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), device=device)
            ], dim=1)
    
    return torch.stack(generated_ids, dim=1) if generated_ids else torch.empty((batch_size, 0), device=device)
