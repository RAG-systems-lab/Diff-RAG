#!/bin/bash
# Diff-RAG 训练脚本
# 依次运行 Stage 2 和 Stage 3/4 训练

set -e

echo "=========================================="
echo "Diff-RAG Training Pipeline"
echo "=========================================="

# 配置
DATA_PATH="${1:-datasets/2wiki/train.json}"  # 默认使用 2wiki 训练集
OUTPUT_BASE="outputs"

# Stage 2: 训练扩散去噪代理
echo ""
echo "=========================================="
echo "Stage 2: Training Diffusion Denoising Agent"
echo "=========================================="
python3 model/experiments/train_stage2.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_BASE/stage2" \
    --epochs 3 \
    --batch_size 32 \
    --lr 1e-4

# Stage 3/4: 训练知识适配器和 LoRA
echo ""
echo "=========================================="
echo "Stage 3 & 4: Training Knowledge Adapter + LoRA"
echo "=========================================="
python3 model/experiments/train_stage34.py \
    --llm_path "Qwen/Qwen2.5-7B-Instruct" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_BASE/stage34" \
    --epochs 3 \
    --batch_size 2 \
    --lr 1e-4 \
    --save_every 1

echo ""
echo "=========================================="
echo "Training Completed!"
echo "=========================================="
echo "Stage 2 output: $OUTPUT_BASE/stage2/relevance_agent.pth"
echo "Stage 3/4 output: $OUTPUT_BASE/stage34/knowledge_adapter.pth"
echo "LoRA output: $OUTPUT_BASE/stage34/qwen_lora_diff_rag/"

