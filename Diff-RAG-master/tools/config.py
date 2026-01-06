class Config:
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    RETRIEVER_NAME = "BAAI/bge-large-en-v1.5"
    DATA_PATH = "./data/2wikimultihop_train.json"
    
    TOP_K_RETRIEVAL = 100
    TOP_K_FINAL = 5
    DIFFUSION_STEPS = 10
    
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    MAX_LENGTH = 32768  
    SEED = 42

cfg = Config()