import json
import random
import torch
import numpy as np

def set_seed(seed=42):
                            
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def build_prompt(query, evidence_texts):
           
    context = "\n".join([f"Doc {i+1}: {txt}" for i, txt in enumerate(evidence_texts)])
    
    prompt = (
        f"Refer to the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    return prompt