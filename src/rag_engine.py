from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from src.config import settings
from src.local_model import load_local_llm,load_embedding_model

class LocalRAGEngine:
    def __init__(self):
        # 1.加载 Embedding, 用于检索问题
        self.embeddings = load_embedding_model()

        # 2.链接向量库
        self.vectorstore = Chroma(
            persist_directory = setting.DB_DIR,
            embedding_function = self.embeddings
        )

        # 3.加载本地LLM
        self.llm = load_local_llm()
    
    def get_qa_chain(self):
        
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
        result = chain.invoke({"query":query})

        return {
            "answer":result['result'],
            "sources":[doc.metadata.get('source') for doc in result['source_documents']]
        }