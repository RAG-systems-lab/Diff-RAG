#!/bin/bash
# ============================================
# Diff-RAG Dataset Download Script
# ============================================

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Diff-RAG Dataset Downloader"
echo "=========================================="
echo ""

# -------------------- 2WikiMultihopQA --------------------
echo "[1/3] Downloading 2WikiMultihopQA..."
mkdir -p 2wiki
cd 2wiki

if [ ! -f "train.json" ]; then
    echo "  Downloading train.json..."
    wget -q --show-progress https://github.com/Alab-NII/2wikimultihop/raw/master/data/train.json || \
    curl -L -o train.json https://github.com/Alab-NII/2wikimultihop/raw/master/data/train.json
else
    echo "  train.json already exists, skipping..."
fi

if [ ! -f "dev.json" ]; then
    echo "  Downloading dev.json..."
    wget -q --show-progress https://github.com/Alab-NII/2wikimultihop/raw/master/data/dev.json || \
    curl -L -o dev.json https://github.com/Alab-NII/2wikimultihop/raw/master/data/dev.json
else
    echo "  dev.json already exists, skipping..."
fi

if [ ! -f "test.json" ]; then
    echo "  Downloading test.json..."
    wget -q --show-progress https://github.com/Alab-NII/2wikimultihop/raw/master/data/test.json || \
    curl -L -o test.json https://github.com/Alab-NII/2wikimultihop/raw/master/data/test.json
else
    echo "  test.json already exists, skipping..."
fi

cd ..
echo "  ✓ 2WikiMultihopQA downloaded successfully"
echo ""

# -------------------- HotpotQA --------------------
echo "[2/3] Downloading HotpotQA..."
mkdir -p hotpotqa
cd hotpotqa

if [ ! -f "hotpot_train_v1.1.json" ]; then
    echo "  Downloading hotpot_train_v1.1.json (large file, ~500MB)..."
    wget -q --show-progress http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json || \
    curl -L -o hotpot_train_v1.1.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
else
    echo "  hotpot_train_v1.1.json already exists, skipping..."
fi

if [ ! -f "hotpot_dev_distractor_v1.json" ]; then
    echo "  Downloading hotpot_dev_distractor_v1.json..."
    wget -q --show-progress http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json || \
    curl -L -o hotpot_dev_distractor_v1.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
else
    echo "  hotpot_dev_distractor_v1.json already exists, skipping..."
fi

cd ..
echo "  ✓ HotpotQA downloaded successfully"
echo ""

# -------------------- PopQA --------------------
echo "[3/3] Downloading PopQA..."
mkdir -p popqa
cd popqa

if [ ! -f "popqa.tsv" ]; then
    echo "  Downloading popqa.tsv..."
    wget -q --show-progress https://github.com/AlexTMallen/adaptive-retrieval/raw/main/data/popqa.tsv || \
    curl -L -o popqa.tsv https://github.com/AlexTMallen/adaptive-retrieval/raw/main/data/popqa.tsv
else
    echo "  popqa.tsv already exists, skipping..."
fi

cd ..
echo "  ✓ PopQA downloaded successfully"
echo ""

# -------------------- Summary --------------------
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Dataset sizes:"
du -sh 2wiki hotpotqa popqa 2>/dev/null || echo "  (size check failed)"
echo ""
echo "Next steps:"
echo "  1. Verify data integrity: python tools/verify_datasets.py"
echo "  2. Start training: bash model/experiments/run_training.sh"
echo ""

