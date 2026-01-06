

import importlib.util
import os

_module_dir = os.path.dirname(__file__)

def _load_module(module_name, class_names):
    file_path = os.path.join(_module_dir, f"{module_name}.py.py")
    if not os.path.exists(file_path):
        file_path = os.path.join(_module_dir, f"{module_name}.py")
    if os.path.exists(file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for class_name in class_names:
            if hasattr(module, class_name):
                globals()[class_name] = getattr(module, class_name)

_load_module("stage1_retrieval", ["HybridRetriever"])
_load_module("stage2_denoising", ["DiffusionDenoisingKernel"])
_load_module("stage3_injection", ["KnowledgeAdapter", "PathLinearizer", "inject_knowledge"])
_load_module("stage4_generation", ["DiffRAGModel", "create_supervised_attention_mask"])

__all__ = [
    "HybridRetriever",
    "DiffusionDenoisingKernel",
    "KnowledgeAdapter",
    "PathLinearizer",
    "inject_knowledge",
    "DiffRAGModel",
    "create_supervised_attention_mask"
]