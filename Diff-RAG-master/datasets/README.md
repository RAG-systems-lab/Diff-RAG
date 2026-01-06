# Datasets Directory

This directory should contain the benchmark datasets for training and evaluation.

## Supported Datasets

### 1. 2WikiMultihopQA
Multi-hop question answering dataset with 2-hop reasoning chains.

**Download**:
```bash
mkdir -p 2wiki
cd 2wiki
wget https://github.com/Alab-NII/2wikimultihop/raw/master/data/train.json
wget https://github.com/Alab-NII/2wikimultihop/raw/master/data/dev.json
wget https://github.com/Alab-NII/2wikimultihop/raw/master/data/test.json
```

### 2. HotpotQA
Large-scale dataset for diverse, explainable multi-hop reasoning.

**Download**:
```bash
mkdir -p hotpotqa
cd hotpotqa
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json
```

### 3. PopQA
Open-domain question answering with popularity-based analysis.

**Download**:
```bash
mkdir -p popqa
cd popqa
# Download from official source
wget https://github.com/AlexTMallen/adaptive-retrieval/raw/main/data/popqa.tsv
```

## Directory Structure

After downloading, your structure should look like:

```
datasets/
├── 2wiki/
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── hotpotqa/
│   ├── hotpot_train_v1.1.json
│   ├── hotpot_dev_distractor_v1.json
│   └── hotpot_test_fullwiki_v1.json
└── popqa/
    └── popqa.tsv
```

## Automatic Download Script

You can use the provided script to download all datasets:

```bash
bash datasets/download_datasets.sh
```

## Data Format

All datasets are expected in JSON/JSONL format with the following fields:
- `question` or `query`: The question text
- `context`: List of candidate passages
- `supporting_facts`: Golden evidence (for training)
- `answer`: Ground truth answer

## Notes

- **Size**: ~850MB total
- **License**: Please check individual dataset licenses before use
- **Preprocessing**: Raw data will be automatically preprocessed during training

