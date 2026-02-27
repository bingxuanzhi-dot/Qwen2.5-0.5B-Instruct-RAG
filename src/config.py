import os
import torch

class Config:
    DATA_DIR = os.path.join(os.getcwd(),"data") # 获取当前工作目录并拼接 data 得到读取 PDF 的路径
    DB_DIR = os.path.join(os.getcwd(),"db_storage") # 获取当前工作目录并拼接 db_storage 得到生成向量数据库存储的路径

    # 第一次运行会自动从 HuggingFace 下载，约 1GB
    LLM_MODEL_ID = "Qwen/Qwen3-0.6B"

    # 本地 Embedding 模型
    EMBEDDING_MODEL_ID = "sentence-transformers/all_MiniLM-L6-v2"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    CHUNK_SIZE = 500 # 每一个数据块的大小
    CHUNK_OVERLAP = 50 # 每两个数据块之间的重叠部分
    TOP_K = 3 # 每次回答一个问题时，需要去数据库找最相似的 TOP_K 片段，然后把这些数量的片段喂给大模型让它整理答案

settings = Config()
print(f"当前运行设备:{settings.DEVICE}")

