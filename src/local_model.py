import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from src.config import settings

def load_embedding_model():
    """
    加载本地 embedding 模型 MiniLM
    它会把 PDF 切片变成向量存进数据库，然后在提问时，迅速把最相关的片段找出来，递给 Qwen 去读。
    """

    print(f"正在加载 Embedding 模型:{settings.EMBEDDING_MODEL_ID}...")
    embeddings = HuggingFaceEmbeddings(
        model_name = settings.EMBEDDING_MODEL_ID,
        model_kwargs = {'device':settings.DEVICE}
    )
    return embeddings

def load_local_llm():
    """
    加载本地 Qwen 模型并包装成 LangChain 可用的接口
    """
    print(f"正在加载LLM:{settings.LLM_MODEL_ID}...")

    # 1.加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        settings.LLM_MODEL_ID,
        trust_remote_code=True
    )

    # 2.加载模型权重
    model = AutoModelForCausalLM.from_pretrained(
        settings.LLM_MODEL_ID,
        torch_dtype = "auto",
        device_map = settings.DEVICE,
        trust_remote_code = True,
    )

    # 3.创建推理管道
    pipe = pipeline(
        "text-generation",
        model = model,
        tokenizer = tokenizer,
        max_new_tokens = 512, # 最多生成 512 个 token
        temperature = 0.1,
        repetition_penalty = 1.1, # 复读机惩罚
        do_sample = True # 允许采样，让回答有一点点随机性
    )

    # 4.包装成 LangChain 对象
    llm = HuggingFacePipeline(pipeline = pipe)

    return llm