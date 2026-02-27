# Qwen3-0.6B RAG

这是一个基于 **Qwen3-0.6B** 模型与 **LangChain** 框架构建的轻量级 RAG 项目。本项目利用 **ChromaDB** 作为向量数据库，支持对 PDF 文档进行本地知识库检索与问答。

## 🛠️ 环境安装 (Installation)

本项目建议使用 Conda 进行环境管理，Python 版本需为 3.10。

### 1. 创建并激活虚拟环境

```bash
conda create -n qwen3_rag python=3.10
conda activate qwen3_rag

# 安装深度学习与大模型基础库：
pip install torch transformers accelerate sentence-transformers

# 安装 RAG 框架与向量数据库组件：
pip install langchain langchain-community langchain-huggingface chromadb pypdf
```
