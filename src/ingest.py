import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.config import settings
from src.local_model import load_embedding_model

def ingest():
    """
    这个过程相当于“知识入库过程”
    如果说 RAG 系统是一个“图书馆”，那么这个脚本就是 “图书管理员进货和上架” 的过程。它负责把在 data 文件夹里放的 PDF 书籍，拆解、加工、翻译成机器能懂的索引，最后整齐地摆放在书架（向量数据库）上。
    """
    # 1.检查文件
    if not os.path.exists(settings.DATA_DIR):
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
        docs.extend(loader.load()) # 每个文件一页一页载入
    
    # 3.切分文本
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = settings.CHUNK_SIZE,
        chunk_overlap = settings.CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs) # 每个片段既包含文字，也包含元数据（属于哪一页）。

    # 4.本地向量化存储
    print("正在生成向量")
    embedding_model = load_embedding_model() # MiniLM 模型

    if os.path.exists(settings.DB_DIR): # 删除之前已有的数据库
        shutil.rmtree(settings.DB_DIR)
    
    Chroma.from_documents(
        documents = splits,
        embedding = embedding_model,
        persist_directory = settings.DB_DIR
    ) # 拿 splits 里的每一块，丢给 embedding_model，算出一串向量（数字列表）。
    print(f"完成！共存入 {len(splits)} 个片段。")

if __name__=="__main__":
    ingest()

