## Diff-RAG

Diff-RAG is a 4-stage RAG pipeline: **Hybrid Retrieval → Diffusion Denoising → Knowledge Injection → Cache-accelerated Generation**.

### Requirements

- **Python**: 3.10+
- **GPU (recommended)**: NVIDIA A100-class or better
- **PyTorch**: CUDA build recommended

### Installation

```bash
git clone https://github.com/RAG-systems-lab/Diff-RAG.git
cd Diff-RAG

# (optional) create env
conda create -n diffrag python=3.10 -y
conda activate diffrag

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Models (optional local download)

- **Dense/KG encoder**: [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)
- **Base LLM**: [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

If you download to local folders, the code will automatically prefer:
- `LLM/BAAI/bge-large-en-v1.5`
- `LLM/Qwen/Qwen2.5-7B-Instruct`

### Quickstart (Inference)

Run with **BF16 (recommended on A100)**:

```bash
python main.py \
  --relevance_agent_path <path_to_stage2_relevance_agent.pth> \
  --adapter_path <path_to_stage34_knowledge_adapter.pth> \
  --data_path datasets/2wiki/test.jsonl \
  --output_file outputs/predictions/result.jsonl \
  --quantization None \
  --dtype bf16 \
  --enable_kv_cache \
  --diffusion_steps 10 \
  --top_k_retrieval 100 \
  --final_k_min 3 \
  --final_k_max 10 \
  --final_k_cumprob 0.8
```

Measure per-stage latency:

```bash
python main.py \
  --relevance_agent_path <path> \
  --adapter_path <path> \
  --data_path datasets/2wiki/test.jsonl \
  --output_file outputs/predictions/result.jsonl \
  --quantization None \
  --dtype bf16 \
  --enable_kv_cache \
  --measure_latency
```

Optional INT8/INT4 quantization (production-oriented; requires `bitsandbytes`):

```bash
pip install bitsandbytes

python main.py --quantization int8 --enable_kv_cache --relevance_agent_path <path> --adapter_path <path>
python main.py --quantization int4 --enable_kv_cache --relevance_agent_path <path> --adapter_path <path>
```

### Training

Stage 2 (RelevanceAgent):

```bash
python model/experiments/train_stage2.py \
  --data_path datasets/2wiki/train.json \
  --output_dir outputs/stage2 \
  --epochs 3
```

Stage 3/4 (Knowledge Adapter + LoRA):

```bash
python model/experiments/train_stage34.py \
  --llm_path Qwen/Qwen2.5-7B-Instruct \
  --data_path datasets/2wiki/train.json \
  --output_dir outputs/stage34 \
  --epochs 3
```

### Notes

- **spaCy model**: `en_core_web_sm` must be installed via `python -m spacy download en_core_web_sm`.
- **Long context (32k)**: the project config sets max length to 32768 (see `tools/config.py`). Your base LLM must support long context.
- **No vLLM**: inference uses **Transformers + custom `SRKIQwen2ForCausalLM`** for knowledge injection; vLLM is not integrated.
- **Fair comparison**: for research/evaluation, keep `--quantization None` and report `--dtype` + hardware.

### License

MIT License. See `LICENSE`.


