# Qwen2.5-0.5B-Instruct-RAG

这是一个基于 **Qwen2.5-0.5B-Instruct** 模型与 **LangChain** 框架构建的轻量级 RAG 项目。

本项目利用 **ChromaDB** 作为向量数据库，支持对 PDF 文档进行本地知识库检索与问答。

## 🛠️ 环境安装 (Installation)

本项目建议使用 Conda 进行环境管理，Python 版本需为 3.10。

### 创建并激活虚拟环境

```bash
conda create -n qwen2.5_rag python=3.10
conda activate qwen2.5_rag

# 安装深度学习与大模型基础库：
pip install torch torchvision torchaudio transformers==4.42.4 accelerate==0.32.1 sentence-transformers==3.0.1

# 安装 RAG 框架与向量数据库组件：
pip install langchain==0.2.14 langchain-community==0.2.12 langchain-core==0.2.33 langchain-huggingface==0.0.3 langchain-chroma==0.1.2 chromadb==0.5.5 huggingface-hub==0.23.4 pypdf==4.3.1   
```


## 🚀 快速开始 (Quick Start)

### 1.准备数据
将收集或者生成的数据以 PDF 格式存放在 data/ 目录下, 本项目包括 [xiaohong.pdf](https://github.com/bingxuanzhi-dot/Qwen2.5-0.5B-Instruct-RAG/blob/main/data/xiaohong.pdf)、[xiaoli.pdf](https://github.com/bingxuanzhi-dot/Qwen2.5-0.5B-Instruct-RAG/blob/main/data/xiaoli.pdf) 和 [xiaoming.pdf](https://github.com/bingxuanzhi-dot/Qwen2.5-0.5B-Instruct-RAG/blob/main/data/xiaoming.pdf)。


前往 [Qwen2.5-0.5B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct)下载模型权重并保存到主目录下。

### 2.运行推理
直接运行主程序即可：
```bash
python train.py
```


## 📂 文件说明 (File Structure)

| 路径/文件 | 类型 | 说明 |
| :--- | :---: | :--- |
| `data/` | 📁 目录 | **[输入]** 存放 PDF 文档 |
| `db_storage/` | 📁 目录 | **[输出]** 向量数据库存储目录 |
| `src/config.py` | 🐍 脚本 | **全局配置**：模型路径、切片参数、设备选择 |
| `src/ingest.py` | 🐍 脚本 | **知识库构建**：ETL 流程 (加载 -> 切分 -> 向量化 -> 存储) |
| `src/local_model.py` | 🐍 脚本 | **模型加载**：负责加载 Qwen 和 Embedding 模型 |
| `src/rag_engine.py` | 🐍 脚本 | **RAG 引擎**：组装检索链、Prompt 模板 |
| `main.py` | 🚀 入口 | **启动脚本**：主程序运行入口 |


## 🙏 致谢与引用 (Acknowledgement)

本项目基于以下优秀的开源项目构建，特此感谢：

*   **[Qwen2.5 (通义千问)](https://github.com/QwenLM/Qwen2.5)** - 阿里云开源的高性能 LLM，提供了强大的推理能力。
*   **[LangChain](https://github.com/langchain-ai/langchain)** - 用于构建大模型应用的顶级框架，简化了 RAG 流程。
*   **[ChromaDB](https://github.com/chroma-core/chroma)** - 轻量级且高效的向量数据库，支持本地持久化存储。
*   **[Sentence Transformers](https://sbert.net/)** - 提供了高质量的 `all-MiniLM-L6-v2` 嵌入模型。
*   **[Hugging Face](https://huggingface.co/)** - 提供了丰富的模型托管与 Transformers 库支持。

特别感谢 **开源社区** 的贡献者们！🚀


## 📄 License
MIT License
