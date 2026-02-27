import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.config import settings
from src.local_model import load_embedding_model

def ingest():
    # 1.检查文件
    if not os.path.exist(settings.DATA_DIR):
        os.makedirs(settings.DATA_DIR)
        print("请在data文件夹放入PDF文件")
        return
    
    files = [f for f in os.listdir(settings.DATA_DIR) if f.endswith('.pdf')]
    if not files:
        print("data目录下没有PDF文件")
        return
    
    # 2.加载 PDF
    docs = []
    for f in files:
        print(f"加载文件:{f}")
        loader = PyPDFLoader(os.path.join(settings.DATA_DIR, f))
        docs.extend(loader.load())
    
    # 3.切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = settings.CHUNK_SIZE,
        chunk_overlap = settings.CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)

    # 4.本地向量化存储
    print("正在生成向量")
    embedding_model = load_embedding_model()

    if os.path.exists(settings.DB_DIR):
        shutil.rmtree(settings.DB_DIR)
    
    Chroma.from_documents(
        documents = splits,
        embedding = embedding_model,
        persist_directory = settings.DB_DIR
    )
    print(f"完成！共存入 {len(splits)} 个片段。")

if __name__=="__main__":
    ingest()

