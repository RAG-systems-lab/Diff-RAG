import collections
import string
import re
import numpy as np

def normalize_answer(s):
           
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common_tokens.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_substring_match(prediction, ground_truth):
           
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)
    if not norm_gt:
        return 0
    return int(norm_gt in norm_pred)

def get_max_metrics(prediction, ground_truths, dataset_type="qa"):
           
    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths]
    
    em_scores = [compute_exact_match(prediction, gt) for gt in ground_truths]
    f1_scores = [compute_f1(prediction, gt) for gt in ground_truths]
    
    metrics = {
        "EM": max(em_scores) if em_scores else 0,
        "F1": max(f1_scores) if f1_scores else 0
    }
    
    if dataset_type.lower() == "popqa":
        acc_scores = [compute_substring_match(prediction, gt) for gt in ground_truths]
        metrics["Accuracy"] = max(acc_scores) if acc_scores else 0
        
    return metrics

def compute_retrieval_metrics(retrieved_ids, gold_ids, k_list=[5, 10]):
           
    metrics = {}
    gold_ids_set = set(gold_ids)
    n_gold = len(gold_ids_set)
    
    if n_gold == 0:
        for k in k_list:
            metrics[f'Recall@{k}'] = 0.0
            metrics[f'Precision@{k}'] = 0.0
        return metrics

    for k in k_list:
        current_top_k = retrieved_ids[:k]
        top_k_set = set(current_top_k)
        
        num_hit = len(top_k_set.intersection(gold_ids_set))
        
        metrics[f'Recall@{k}'] = num_hit / n_gold
        
        metrics[f'Precision@{k}'] = num_hit / k
        
    return metrics

def count_tokens(text):
           
    if not text: 
        return 0
    return len(text.split())

class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.scores = collections.defaultdict(list)
        self.count = 0

    def update(self, metrics_dict):
        for k, v in metrics_dict.items():
            self.scores[k].append(v)
        self.count += 1

    def get_avg(self):
        return {k: sum(v)/len(v) for k, v in self.scores.items()}

    def summary(self):
        avg = self.get_avg()
        priority = ['Accuracy', 'F1', 'EM', 'Recall@10', 'Precision@10', 'Tokens']
        res = []
        
        for k in priority:
            if k in avg:
                res.append(f"{k}: {avg[k]:.4f}")
        
        for k, v in avg.items():
            if k not in priority:
                res.append(f"{k}: {v:.4f}")
                
        return " | ".join(res)