
import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from sentence_transformers import SentenceTransformer
import json
import argparse
from tqdm import tqdm
from tools.data_utils import load_jsonl, set_seed

class KnowledgeAdapter(nn.Module):
    def __init__(self, encoder_dim=1024, llm_dim=3584):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, llm_dim),
            nn.SiLU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, x):
        return self.projector(x)

class DiffRAGModel(nn.Module):
    def __init__(self, llm_path, adapter_path=None, device="cuda"):
        super().__init__()
        self.device = device
        
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        local_model_paths = [
            os.path.join(_project_root, "LLM", "Qwen2.5-7B-Instruct"),
            os.path.join(_project_root, "LLM", "Qwen", "Qwen2.5-7B-Instruct"),
            os.path.join(_project_root, "LLM", llm_path.replace("/", "_")),
        ]
        
        actual_llm_path = llm_path
        for local_path in local_model_paths:
            if os.path.exists(local_path):
                print(f"Using local LLM: {local_path}")
                actual_llm_path = local_path
                break
        else:
            print(f"Loading LLM from {llm_path} (will use cache or download if available)...")
        
        is_local_path = os.path.isabs(actual_llm_path) or "snapshots" in actual_llm_path or os.path.exists(actual_llm_path)
        use_local_only = is_local_path
        
        print(f"Loading LLM from: {actual_llm_path}")
        print(f"  local_files_only: {use_local_only}")
        
        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                actual_llm_path, 
                dtype=torch.bfloat16, 
                trust_remote_code=True,
                device_map="auto" if device == "cuda" else None,
                local_files_only=use_local_only
            )
        except Exception as e:
            if use_local_only:
                print(f"Warning: local_files_only=True failed, trying with False: {e}")
                self.llm = AutoModelForCausalLM.from_pretrained(
                    actual_llm_path, 
                    dtype=torch.bfloat16, 
                    trust_remote_code=True,
                    device_map="auto" if device == "cuda" else None,
                    local_files_only=False
                )
            else:
                raise
        if device == "cpu":
            self.llm = self.llm.to(device)
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.print_trainable_parameters()
        
        llm_dim = self.llm.config.hidden_size
        self.adapter = KnowledgeAdapter(encoder_dim=1024, llm_dim=llm_dim).to(self.device)
        if adapter_path and os.path.exists(adapter_path):
            self.adapter.load_state_dict(torch.load(adapter_path, map_location=device))
            print(f"Loaded adapter from {adapter_path}")
        
        local_model_path = os.path.join(_project_root, "LLM/BAAI/bge-large-en-v1.5")
        if os.path.exists(local_model_path):
            print(f"  Using local encoder: {local_model_path}")
            self.encoder = SentenceTransformer(local_model_path, device="cpu")
        else:
            print("  Using HuggingFace encoder: BAAI/bge-large-en-v1.5")
            self.encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
        self.encoder.eval()

    def forward(self, query_texts, evidence_texts, answer_texts, tokenizer):
        """前向传播计算 Loss"""
        batch_size = len(query_texts)
        
        with torch.no_grad():
            evidence_embs = self.encoder.encode(evidence_texts, convert_to_tensor=True, show_progress_bar=False)
            evidence_embs = evidence_embs.to(self.device)
            evidence_embs = evidence_embs.unsqueeze(1)

        virtual_token_embeds = self.adapter(evidence_embs.float())

        prompts = [f"Question: {q}\nAnswer:" for q in query_texts]

        prompt_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=32768).input_ids.to(self.device)
        answer_ids = tokenizer(answer_texts, return_tensors="pt", padding=True, truncation=True, max_length=32768).input_ids.to(self.device)
        
        if hasattr(self.llm.model, 'embed_tokens'):
            embed_fn = self.llm.model.embed_tokens
        elif hasattr(self.llm, 'get_input_embeddings'):
            embed_fn = self.llm.get_input_embeddings()
        else:
            embed_fn = self.llm.model.get_input_embeddings()
        
        prompt_embeds = embed_fn(prompt_ids)
        answer_embeds = embed_fn(answer_ids)
        
        inputs_embeds = torch.cat([virtual_token_embeds, prompt_embeds, answer_embeds], dim=1)
        
        virtual_len = virtual_token_embeds.shape[1]
        prompt_len = prompt_ids.shape[1]
        answer_len = answer_ids.shape[1]
        
        ignore_label = -100
        labels = torch.full((batch_size, virtual_len + prompt_len + answer_len), ignore_label, dtype=torch.long).to(self.device)
        labels[:, virtual_len + prompt_len:] = answer_ids

        outputs = self.llm(inputs_embeds=inputs_embeds, labels=labels)
        return outputs.loss

class AlignmentDataset(Dataset):
    def __init__(self, data_path=None):
        if data_path and os.path.exists(data_path):
            self.data = self._load_from_file(data_path)
            print(f"Loaded {len(self.data)} samples from {data_path}")
        else:
            self.data = [
                {
                    "query": "Where is Paris?",
                    "evidence": "Paris is capital of France. [SEP] Struct: Paris linked to Eiffel Tower.",
                    "answer": "Paris is in France."
                },
                {
                    "query": "Who is Musk?",
                    "evidence": "Musk is CEO of Tesla. [SEP] Struct: Musk linked to SpaceX.",
                    "answer": "Elon Musk is a tech entrepreneur."
                }
            ] * 100
            print(f"Using synthetic data (size={len(self.data)})")

    def _load_from_file(self, data_path):
        data = []
        if data_path.endswith('.jsonl'):
            items = load_jsonl(data_path)
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                items = json.load(f)
        
        for item in items:
            query = item.get('question', item.get('query', ''))
            answer = item.get('answer', '')
            
            evidence_parts = []
            if 'context' in item and isinstance(item['context'], list):
                for ctx in item['context'][:3]:
                    if isinstance(ctx, list) and len(ctx) > 1:
                        ctx_text = ctx[1]
                        if isinstance(ctx_text, list):
                            ctx_text = " ".join(str(s) for s in ctx_text)
                        else:
                            ctx_text = str(ctx_text)
                        if ctx_text and len(ctx_text.strip()) > 10:
                            evidence_parts.append(ctx_text)
            evidence = " [SEP] ".join(evidence_parts) if evidence_parts else "No evidence available."
            
            if query and answer:
                data.append({
                    "query": query,
                    "evidence": evidence,
                    "answer": answer if isinstance(answer, str) else str(answer[0]) if isinstance(answer, list) else str(answer)
                })
        
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['query'], item['evidence'], item['answer']

def collate_fn_strings(batch):
   
    queries, evidences, answers = zip(*batch)
    return list(queries), list(evidences), list(answers)

def train_stage34(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")
    
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    if hasattr(args, 'local_llm_path') and args.local_llm_path:
        actual_llm_path = args.local_llm_path
        print(f"Using specified local LLM: {actual_llm_path}")
    else:
        local_model_paths = [
            os.path.join(_project_root, "LLM", "Qwen2.5-7B-Instruct"),
            os.path.join(_project_root, "LLM", "Qwen", "Qwen2.5-7B-Instruct"),
            os.path.join(_project_root, "LLM", args.llm_path.replace("/", "_")),
            os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots"),
        ]
        
        actual_llm_path = args.llm_path
        found_local = False
        for local_path in local_model_paths:
            if os.path.exists(local_path):
                if "snapshots" in local_path:
                    snapshots = [d for d in os.listdir(local_path) if os.path.isdir(os.path.join(local_path, d))]
                    if snapshots:
                        actual_llm_path = os.path.join(local_path, sorted(snapshots)[-1])
                        print(f"Using cached HuggingFace model: {actual_llm_path}")
                        found_local = True
                        break
                else:
                    print(f"Using local LLM: {local_path}")
                    actual_llm_path = local_path
                    found_local = True
                    break
        
        if not found_local:
            print(f"Loading tokenizer from {args.llm_path} (will use cache if available)...")
    
    if hasattr(args, 'local_llm_path') and args.local_llm_path:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                actual_llm_path, 
                trust_remote_code=True, 
                local_files_only=True
            )
        except Exception as e:
            print(f"Warning:   : {e}")
            if "Qwen2.5-7B-Instruct" in actual_llm_path:
                model_name = "Qwen/Qwen2.5-7B-Instruct"
            elif "Qwen3-8B" in actual_llm_path:
                model_name = "Qwen/Qwen3-8B"
            else:
                model_name = args.llm_path
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=False
            )
    else:
        use_local_only = actual_llm_path != args.llm_path
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                actual_llm_path, 
                trust_remote_code=True, 
                local_files_only=use_local_only
            )
        except Exception as e:
            print(f"Warning: local_files_only failed, trying without: {e}")
            tokenizer = AutoTokenizer.from_pretrained(
                actual_llm_path, 
                trust_remote_code=True, 
                local_files_only=False
            )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    final_llm_path = args.local_llm_path if (hasattr(args, 'local_llm_path') and args.local_llm_path) else actual_llm_path
    print(f"Final LLM path for model loading: {final_llm_path}")
    model = DiffRAGModel(final_llm_path, adapter_path=args.adapter_path, device=device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    
    dataset = AlignmentDataset(data_path=args.data_path)
    num_workers = max(16, args.batch_size * 2)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn_strings,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
        prefetch_factor=8 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1), 
        num_training_steps=total_steps
    )
    
    start_epoch = 0
    global_step = 0
    if args.resume_from:
        checkpoint_path = args.resume_from
        if os.path.exists(checkpoint_path):
            print(f"  : {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if isinstance(checkpoint, dict) and 'adapter_state_dict' in checkpoint:
                model.adapter.load_state_dict(checkpoint['adapter_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch']
                if 'global_step' in checkpoint:
                    global_step = checkpoint['global_step']
                print(f"  已恢复: Epoch {start_epoch}, Global Step {global_step}")
            else:
                model.adapter.load_state_dict(checkpoint)
                import re
                match = re.search(r'epoch(\d+)', checkpoint_path)
                if match:
                    start_epoch = int(match.group(1))
                    print(f"  已加载 adapter（旧格式），从 Epoch {start_epoch} 继续训练")
                else:
                    print(f"  已加载 adapter（旧格式），从 Epoch 0 继续训练")
        else:
            print(f"警告: Checkpoint 文件不存在: {checkpoint_path}，从头开始训练")
    
    print(f"Start Training Adapter + LoRA...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Total steps: {total_steps}")
    
    use_amp = device == "cuda" and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("  ✅ 混合精度训练已启用 (AMP)")
    
    model.train()
    
    for epoch in range(start_epoch, args.epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            queries, evidences, answers = batch
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(list(queries), list(evidences), list(answers), tokenizer)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model(list(queries), list(evidences), list(answers), tokenizer)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            
            total_loss += loss.item()
            global_step += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % args.save_every == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            adapter_path = os.path.join(args.output_dir, f"knowledge_adapter_epoch{epoch+1}.pth")
            checkpoint = {
                'adapter_state_dict': model.adapter.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_step': global_step,
                'loss': avg_loss
            }
            torch.save(checkpoint, adapter_path)
            print(f"  Saved checkpoint to {adapter_path}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    adapter_path = os.path.join(args.output_dir, "knowledge_adapter.pth")
    lora_path = os.path.join(args.output_dir, "qwen_lora_diff_rag")
    
    torch.save(model.adapter.state_dict(), adapter_path)
    model.llm.save_pretrained(lora_path)
    print(f"Stage 3 & 4 Training Completed.")
    print(f"  Adapter saved to: {adapter_path}")
    print(f"  LoRA saved to: {lora_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 3 & 4: Knowledge Adapter + LoRA")
    parser.add_argument("--llm_path", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Path to base LLM (HuggingFace model name)")
    parser.add_argument("--local_llm_path", type=str, default=None, help="Local path to LLM model (overrides llm_path)")
    parser.add_argument("--data_path", type=str, default=None, help="Path to training data (JSON/JSONL)")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to existing adapter (optional)")
    parser.add_argument("--output_dir", type=str, default="outputs/stage34", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (optimized for 80GB A100, increased for speed)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume training from checkpoint (e.g., outputs/stage34/knowledge_adapter_epoch2.pth)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (even if CUDA available)")
    
    args = parser.parse_args()
    train_stage34(args)

