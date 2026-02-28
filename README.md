# Qwen2.5-0.5B-Instruct-RAG

这是一个基于 **Qwen2.5-0.5B-Instruct** 模型与 **LangChain** 框架构建的轻量级 RAG 项目。本项目利用 **ChromaDB** 作为向量数据库，支持对 PDF 文档进行本地知识库检索与问答。

## 🛠️ 环境安装 (Installation)

本项目建议使用 Conda 进行环境管理，Python 版本需为 3.10。

### 1. 创建并激活虚拟环境

```bash
conda create -n qwen2.5_rag python=3.10
conda activate qwen2.5_rag

# 安装深度学习与大模型基础库：
pip install torch torchvision torchaudio transformers==4.42.4 accelerate==0.32.1 sentence-transformers==3.0.1

# 安装 RAG 框架与向量数据库组件：
pip install langchain==0.2.14 langchain-community==0.2.12 langchain-core==0.2.33 langchain-huggingface==0.0.3 langchain-chroma==0.1.2 chromadb==0.5.5 huggingface-hub==0.23.4 pypdf==4.3.1   
```
