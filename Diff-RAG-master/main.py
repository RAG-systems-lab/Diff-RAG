import argparse
import torch
import torch.nn.functional as F
import json
import os
import time
from tqdm import tqdm
from typing import List, Optional, Tuple

from model.methodology.stage1_retrieval import HybridRetriever
from model.methodology.stage2_denoising import DiffusionDenoisingKernel
from model.methodology.stage3_injection import KBEncoder, KnowledgeAdapter
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
try:
    from transformers import BitsAndBytesConfig
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    print("Warning: bitsandbytes not installed. Quantization will not work. Install with: pip install bitsandbytes")

import importlib.util
_qwen_model_path = os.path.join(os.path.dirname(__file__), 'model', 'methodology', 'Qwen2.5-7B-Instruct.py')
spec = importlib.util.spec_from_file_location("qwen_model", _qwen_model_path)
qwen_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qwen_model)
SRKIQwen2ForCausalLM = qwen_model.SRKIQwen2ForCausalLM

from tools.config import cfg
from tools.metrics import MetricTracker, get_max_metrics, compute_retrieval_metrics, count_tokens
from tools.data_utils import load_jsonl, save_jsonl, set_seed


class OptimizedDiffRAGPipeline:
    
    def __init__(self, llm_path: str, adapter_path: Optional[str] = None, 
                 relevance_agent_path: Optional[str] = None,
                 device: str = "cuda",
                 retrieval_layer_idx: int = 16,
                 enable_kv_cache: bool = True,
                 enable_compile: bool = False,
                 quantization: Optional[str] = None,
                 dtype: str = "bf16"):
        self.device = device
        self.retrieval_layer_idx = retrieval_layer_idx
        self.enable_kv_cache = enable_kv_cache
        self.quantization = quantization  # "int8", "int4", "fp8", None
        self.dtype = dtype  # "bf16", "fp16", "fp32"
        
        print(">>> [Stage 1 Init] Loading Hybrid Retriever...")
        self.retriever = HybridRetriever(device=device)
        
        print(">>> [Stage 2 Init] Loading Diffusion Denoising Kernel...")
        self.denoiser = DiffusionDenoisingKernel(
            device=device,
            relevance_agent_path=relevance_agent_path
        )
        
        print(">>> [Stage 3 Init] Loading KB Encoder and Knowledge Adapter...")
        self.kb_encoder = KBEncoder(device=device)
        
        _project_root = os.path.abspath(os.path.dirname(__file__))
        local_model_paths = [
            os.path.join(_project_root, "LLM", "Qwen2.5-7B-Instruct"),
            os.path.join(_project_root, "LLM", "Qwen", "Qwen2.5-7B-Instruct"),
            os.path.join(_project_root, "LLM", llm_path.replace("/", "_")),
        ]
        actual_llm_path = llm_path
        for local_path in local_model_paths:
            if os.path.exists(local_path):
                actual_llm_path = local_path
                break
        
        print(f">>> [Stage 4 Init] Loading LLM from {actual_llm_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            actual_llm_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        config = AutoConfig.from_pretrained(actual_llm_path, trust_remote_code=True)
        
        if not hasattr(config, 'knowledge_info') or config.knowledge_info is None:
            config.knowledge_info = [retrieval_layer_idx]
            config.kb_layer_frequency = 1
        if not hasattr(config, 'sep_query_head'):
            config.sep_query_head = True
        if not hasattr(config, 'project_type'):
            config.project_type = "linear"
        if not hasattr(config, 'embed_dim'):
            config.embed_dim = 1024
        if not hasattr(config, 'top_k_kb'):
            config.top_k_kb = 20
        if not hasattr(config, 'dynamic_sparsify'):
            config.dynamic_sparsify = True
        if not hasattr(config, 'return_retr_logits'):
            config.return_retr_logits = False

        # Precision for non-quantized weights (A100 recommended: bf16)
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        if self.dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {self.dtype}. Choose from {list(dtype_map.keys())}.")
        base_torch_dtype = dtype_map[self.dtype]

        if quantization == "int8":
            if not HAS_BITSANDBYTES:
                raise ImportError("bitsandbytes is required for INT8 quantization. Install with: pip install bitsandbytes")
            print(">>> Loading model with INT8 quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            self.llm = SRKIQwen2ForCausalLM.from_pretrained(
                actual_llm_path,
                config=config,
                quantization_config=quantization_config,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
        elif quantization == "int4":
            if not HAS_BITSANDBYTES:
                raise ImportError("bitsandbytes is required for INT4 quantization. Install with: pip install bitsandbytes")
            print(">>> Loading model with INT4 quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=base_torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.llm = SRKIQwen2ForCausalLM.from_pretrained(
                actual_llm_path,
                config=config,
                quantization_config=quantization_config,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
        elif quantization == "bf8" or quantization == "bfloat8":
            print(">>> Loading model with BF8 (bfloat8) quantization...")
            if hasattr(torch, 'float8_e5m2'):
                print("  Using torch.float8_e5m2 (FP8 E5M2 format)")
                self.llm = SRKIQwen2ForCausalLM.from_pretrained(
                    actual_llm_path,
                    config=config,
                    torch_dtype=torch.float8_e5m2,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True
                )
            else:
                print("  Warning: float8_e5m2 not available, falling back to bfloat16")
                print("  Note: BF8 requires PyTorch 2.1+ and compatible hardware (H100/Ada)")
                self.llm = SRKIQwen2ForCausalLM.from_pretrained(
                    actual_llm_path,
                    config=config,
                    torch_dtype=torch.bfloat16,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True
                )
        elif quantization == "fp8":
            print(">>> Loading model with FP8 quantization...")
            self.llm = SRKIQwen2ForCausalLM.from_pretrained(
                actual_llm_path,
                config=config,
                torch_dtype=torch.float8_e5m2 if hasattr(torch, 'float8_e5m2') else torch.bfloat16,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
        else:
            print(f">>> Loading model without quantization ({self.dtype})...")
            self.llm = SRKIQwen2ForCausalLM.from_pretrained(
                actual_llm_path,
                config=config,
                torch_dtype=base_torch_dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            if device == "cpu":
                self.llm = self.llm.to(device)
        

        if enable_compile and hasattr(torch, 'compile'):
            print(">>> Compiling model with torch.compile...")
            self.llm = torch.compile(self.llm, mode="reduce-overhead")
        
        llm_dim = self.llm.config.hidden_size
        self.knowledge_adapter = KnowledgeAdapter(encoder_dim=1024, llm_dim=llm_dim).to(device)
        if adapter_path and os.path.exists(adapter_path):
            self.knowledge_adapter.load_state_dict(torch.load(adapter_path, map_location=device))
            print(f"  Loaded Knowledge Adapter from {adapter_path}")
        
        self.knowledge_adapter.eval()
        self.llm.eval()
        
        print(">>> Pipeline initialization complete!")
    
    def run_stage1_hybrid_retrieval(self, query: str, conversation_history: Optional[List[str]] = None,
                                    corpus: Optional[List[str]] = None, top_k: int = 100) -> Tuple[List[str], torch.Tensor, any]:
        
        candidates, candidate_embs, local_graph = self.retriever.search(
            query=query,
            corpus=corpus,
            top_k=top_k,
            conversation_history=conversation_history
        )
        return candidates, candidate_embs, local_graph
    
    def run_stage2_diffusion_denoising(
        self,
        query: str,
        candidates: List[str],
        candidate_embs: torch.Tensor,
        T: int = 10,
        final_k_min: int = 3,
        final_k_max: int = 10,
        final_k_cumprob: float = 0.8,
    ) -> List[str]:
       
        query_emb = self.retriever.encoder.encode([query], convert_to_tensor=True).to(self.device)
        
        golden_indices = self.denoiser.run_diffusion(
            query_emb,
            candidate_embs,
            T=T,
            final_k_min=final_k_min,
            final_k_max=final_k_max,
            final_cumprob=final_k_cumprob,
        )
        
        if isinstance(golden_indices, torch.Tensor):
            golden_indices = golden_indices.cpu().numpy().tolist()
        elif not isinstance(golden_indices, list):
            golden_indices = golden_indices.tolist() if hasattr(golden_indices, 'tolist') else list(golden_indices)
        
        golden_evidence = [candidates[i] for i in golden_indices]
        return golden_evidence
    
    def run_stage3_knowledge_injection(self, query: str, golden_evidence: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 query_embeds, key_embds, value_embds, query_ids（用于后续生成）"""
        query_ids = self.tokenizer(query, return_tensors="pt").input_ids.to(self.device)
        
        if hasattr(self.llm, 'get_input_embeddings'):
            embed_fn = self.llm.get_input_embeddings()
        elif hasattr(self.llm, 'model') and hasattr(self.llm.model, 'embed_tokens'):
            embed_fn = self.llm.model.embed_tokens
        else:
            embed_fn = self.llm.model.get_input_embeddings()
        
        query_embeds = embed_fn(query_ids)
        
        evidence_embs = self.kb_encoder.encode(golden_evidence, convert_to_tensor=True)
        batch_size = query_embeds.shape[0]
        
        if evidence_embs.dim() == 2:
            evidence_embs = evidence_embs.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            evidence_embs = evidence_embs.expand(batch_size, -1, -1)
        
        key_embds = evidence_embs.float()
        value_embds = evidence_embs.float()
        
        return query_embeds, key_embds, value_embds, query_ids
    
    def run_stage4_supervised_generation(self, query_embeds: torch.Tensor,
                                        key_embds: torch.Tensor,
                                        value_embds: torch.Tensor,
                                        query_ids: torch.Tensor,
                                        max_new_tokens: int = 50,  
                                        sampling_method: str = "nucleus", 
                                        top_p: float = 0.9,
                                        temperature: float = 1.0) -> str:
       
        batch_size = query_embeds.shape[0]
        seq_len = query_embeds.shape[1]
        
        attention_mask = torch.ones((batch_size, seq_len), device=self.device, dtype=torch.long)
        
        generated_ids = []
        current_input = query_embeds
        past_key_values = None  # KV Cache
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                outputs = self.llm(
                    inputs_embeds=current_input if step == 0 else None,
                    input_ids=None if step == 0 else None,
                    attention_mask=attention_mask,
                    key_embds=key_embds if step == 0 else None,  
                    value_embds=value_embds if step == 0 else None,
                    use_cache=self.enable_kv_cache,
                    past_key_values=past_key_values
                )
                
                if self.enable_kv_cache and hasattr(outputs, 'past_key_values'):
                    past_key_values = outputs.past_key_values
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits[:, -1, :]
                else:
                    logits = outputs[:, -1, :]
                
           
                if sampling_method == "nucleus":
                    from model.methodology.stage4_generation import nucleus_sampling
                    next_token = nucleus_sampling(logits, top_p=top_p, temperature=temperature)
                else:
                    next_token = torch.argmax(logits, dim=-1)
                
                generated_ids.append(next_token)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
         
                if step == 0:
               
                    if hasattr(self.llm, 'get_input_embeddings'):
                        embed_fn = self.llm.get_input_embeddings()
                    elif hasattr(self.llm, 'model') and hasattr(self.llm.model, 'embed_tokens'):
                        embed_fn = self.llm.model.embed_tokens
                    else:
                        embed_fn = self.llm.model.get_input_embeddings()
                    
                    next_token_embed = embed_fn(next_token.unsqueeze(1))
                    current_input = torch.cat([current_input, next_token_embed], dim=1)
                else:
                 
                    current_input = next_token.unsqueeze(1)
                
                new_mask = torch.ones((batch_size, 1), device=self.device, dtype=torch.long)
                attention_mask = torch.cat([attention_mask, new_mask], dim=1)
        
        if generated_ids:
            generated_ids_tensor = torch.stack(generated_ids, dim=1)
            pred_text = self.tokenizer.decode(generated_ids_tensor[0], skip_special_tokens=True)
        else:
            pred_text = ""
        
        return pred_text
    
    def inference(
        self,
        query: str,
        conversation_history: Optional[List[str]] = None,
        corpus: Optional[List[str]] = None,
        top_k_retrieval: int = 100,
        diffusion_steps: int = 10,
        final_k_min: int = 3,
        final_k_max: int = 10,
        final_k_cumprob: float = 0.8,
        max_new_tokens: int = 50,
        sampling_method: str = "nucleus",
        measure_latency: bool = False,
    ) -> Tuple[str, List[str], Optional[dict]]:
       
        latency_info = {}
        
        if measure_latency:
            start_time = time.time()
         # Stage 1
        if measure_latency:
            stage1_start = time.time()
        candidates, candidate_embs, local_graph = self.run_stage1_hybrid_retrieval(
            query=query,
            conversation_history=conversation_history,
            corpus=corpus,
            top_k=top_k_retrieval
        )
        if measure_latency:
            latency_info['stage1_retrieval'] = time.time() - stage1_start
        
        # Stage 2: Diffusion Denoising
        if measure_latency:
            stage2_start = time.time()
        golden_evidence = self.run_stage2_diffusion_denoising(
            query=query,
            candidates=candidates,
            candidate_embs=candidate_embs,
            T=diffusion_steps,
            final_k_min=final_k_min,
            final_k_max=final_k_max,
            final_k_cumprob=final_k_cumprob,
        )
        if measure_latency:
            latency_info['stage2_denoising'] = time.time() - stage2_start
        
        # Stage 3: Knowledge Injection
        if measure_latency:
            stage3_start = time.time()
        query_embeds, key_embds, value_embds, query_ids = self.run_stage3_knowledge_injection(
            query=query,
            golden_evidence=golden_evidence
        )
        if measure_latency:
            latency_info['stage3_injection'] = time.time() - stage3_start
        
        # Stage 4: Generation
        if measure_latency:
            stage4_start = time.time()
        answer = self.run_stage4_supervised_generation(
            query_embeds=query_embeds,
            key_embds=key_embds,
            value_embds=value_embds,
            query_ids=query_ids,
            max_new_tokens=max_new_tokens,
            sampling_method=sampling_method
        )
        if measure_latency:
            latency_info['stage4_generation'] = time.time() - stage4_start
            latency_info['total'] = time.time() - start_time
        
        return answer, golden_evidence, (latency_info if measure_latency else None)


def main():
    parser = argparse.ArgumentParser(
        description="Diff-RAG Inference Pipeline with Optional Quantization",
        epilog="""
Examples:
  # For fair comparison (full precision, default)
  python main.py --relevance_agent_path <path> --adapter_path <path> --quantization None
  
  # For production optimization (with quantization)
  python main.py --relevance_agent_path <path> --adapter_path <path> --quantization int8 --enable_kv_cache
        """
    )
    parser.add_argument("--dataset", type=str, default="2wiki", choices=["2wiki", "hotpotqa", "popqa"])
    parser.add_argument("--data_path", type=str, default="datasets/2wiki/test.jsonl")
    parser.add_argument("--output_file", type=str, default="outputs/predictions/result_optimized.jsonl")
    parser.add_argument("--llm_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--relevance_agent_path", type=str, required=True)
    parser.add_argument("--top_k_retrieval", type=int, default=100, help="Top-K retrieval count (default: 100)")
    parser.add_argument("--diffusion_steps", type=int, default=10, help="Diffusion denoising steps (default: 10)")
    parser.add_argument("--final_k_min", type=int, default=3, help="Final evidence k min (default: 3)")
    parser.add_argument("--final_k_max", type=int, default=10, help="Final evidence k max (default: 10)")
    parser.add_argument("--final_k_cumprob", type=float, default=0.8, help="Final evidence selection cumulative probability threshold (default: 0.8)")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum generation tokens (default: 50)")
    parser.add_argument("--sampling_method", type=str, default="nucleus", choices=["nucleus", "greedy"])
    parser.add_argument("--quantization", type=str, default=None, choices=["int8", "int4", "bf8", "bfloat8", "fp8", None], 
                        help="Quantization method: None (default, bfloat16 for fair comparison), bf8/bfloat8 (FP8 E5M2, requires H100/Ada), int8 (for speed), int4 (faster but lower quality), fp8 (experimental)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
                        help="Base precision for non-quantized weights (default: bf16). A100 recommended: bf16.")
    parser.add_argument("--enable_kv_cache", action="store_true", default=True, help="Enable KV cache")
    parser.add_argument("--enable_compile", action="store_true", help="Enable torch.compile (PyTorch 2.0+)")
    parser.add_argument("--measure_latency", action="store_true", help="Measure and report latency")
    args = parser.parse_args()
    
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> Running Optimized Diff-RAG Pipeline on {device}")
    print("=" * 70)
    
    pipeline = OptimizedDiffRAGPipeline(
        llm_path=args.llm_path,
        adapter_path=args.adapter_path,
        relevance_agent_path=args.relevance_agent_path,
        device=device,
        enable_kv_cache=args.enable_kv_cache,
        enable_compile=args.enable_compile,
        quantization=args.quantization,
        dtype=args.dtype,
    )
    
    print(f">>> Loading data from {args.data_path}...")
    dataset = load_jsonl(args.data_path)
    
    tracker = MetricTracker()
    results = []
    total_latency = 0.0
    
    for item in tqdm(dataset, desc="Inference"):
        query = item['question']
        gold_answers = item['answer']
        gold_doc_ids = item.get('supporting_facts_ids', [])
        conversation_history = item.get('conversation_history', None)
        
        answer, golden_evidence, latency_info = pipeline.inference(
            query=query,
            conversation_history=conversation_history,
            corpus=None,
            top_k_retrieval=args.top_k_retrieval,
            diffusion_steps=args.diffusion_steps,
            final_k_min=args.final_k_min,
            final_k_max=args.final_k_max,
            final_k_cumprob=args.final_k_cumprob,
            max_new_tokens=args.max_new_tokens,
            sampling_method=args.sampling_method,
            measure_latency=args.measure_latency
        )
        
        if latency_info:
            total_latency += latency_info['total']
            if len(results) < 5:  
                print(f"\nSample {len(results)+1} Latency Breakdown:")
                for stage, time_ms in latency_info.items():
                    print(f"  {stage}: {time_ms*1000:.2f}ms")
        
        gen_metrics = get_max_metrics(answer, gold_answers, dataset_type=args.dataset)
        retrieved_ids = golden_evidence
        ret_metrics = compute_retrieval_metrics(retrieved_ids, gold_doc_ids, k_list=[5, 10])
        
        num_tokens = count_tokens(query + " " + answer)
        
        step_metrics = {}
        step_metrics.update(gen_metrics)
        step_metrics.update(ret_metrics)
        step_metrics['Tokens'] = num_tokens
        if latency_info:
            step_metrics['Latency_ms'] = latency_info['total'] * 1000
        tracker.update(step_metrics)
        
        results.append({
            "query": query,
            "prediction": answer,
            "golden_evidence": golden_evidence,
            "gold_answers": gold_answers,
            "metrics": step_metrics
        })
    
    print("\n" + "=" * 70)
    print(f"Final Results on {args.dataset}:")
    print(tracker.summary())
    if args.measure_latency:
        avg_latency = total_latency / len(dataset) * 1000
        print(f"\nAverage Latency: {avg_latency:.2f}ms")
        print(f"Target: <450ms | Achieved: {'✅' if avg_latency < 450 else '❌'}")
    print("=" * 70)
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    save_jsonl(results, args.output_file)
    print(f">>> Predictions saved to {args.output_file}")


if __name__ == "__main__":
    main()

