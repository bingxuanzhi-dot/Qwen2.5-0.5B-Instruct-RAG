from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from src.config import settings
from src.local_model import load_local_llm,load_embedding_model

class LocalRAGEngine:
    """
    定义一个类，把 RAG 所有零件封装在一起
    """
    def __init__(self):
        # 1.加载 Embedding。后面连接数据库时，Chroma 需要知道用哪个模型把“问题变成向量。必须和存入数据时用的模型完全一致，否则搜出来的结果是乱码。
        self.embeddings = load_embedding_model()

        # 2.链接向量库
        self.vectorstore = Chroma(
            persist_directory = settings.DB_DIR,
            embedding_function = self.embeddings
        )

        # 3.加载本地LLM
        self.llm = load_local_llm()
    
    def get_qa_chain(self):
        """
        {context} 和 {question} 都是占位符，在程序运行时会分别把检索到的 PDF 文字和用户问题填在这里。
        流程：构造对话模板 -> 构造回答链 -> 定义聊天接口
        """

        template = """使用以下上下文片段来回答最后的问题。
        如果你不知道答案, 只需说不知道, 不要试图编造答案。
        回答要简短。

        上下文: {context}

        问题: {question}

        回答:"""

        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # 构建检索问答链
        qa_chain = RetrievalQA.from_chain_type(
            llm = self.llm,
            chain_type = "stuff",
            retriever = self.vectorstore.as_retriever(search_kwargs={"k":settings.TOP_K}),
            return_source_documents = True,
            chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
        )

        return qa_chain
    
    def chat(self, query):
        chain = self.get_qa_chain()

        """
        下面这一步 result = chain.invoke({"query":query}) 会发生：
        1.Embedding 模型把 query 变成向量。
        2.Chroma 检索出 Top K 文档。
        3.Prompt 模板把 Top K 文档和 query 填空。
        4.Qwen 模型接收 Prompt, 生成回答。
        5.result 拿到了包含 result(答案)和 source_documents(原文)的字典。
        """
        result = chain.invoke({"query":query})

        return {
            "answer":result['result'],
            "sources":[doc.metadata.get('source') for doc in result['source_documents']]
        }