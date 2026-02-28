import os
from src.ingest import ingest
from src.rag_engine import LocalRAGEngine

def main():
    """
    整个 RAG 系统的启动脚本
    任务流程：检查环境 -> 启动引擎 -> 建立对话
    """
    print("=== 本地全栈 RAG 系统 ===")

    # 1.询问是否需要构建知识库
    if not os.path.exists("db_storage"):
        print("检测到初次运行，正在构建知识库...")
        ingest()
    else:
        choice = input("是否更新知识库(y/n)")
        if choice.lower()=='y':
            ingest()
    
    # 2.初始化引擎
    print("\n正在启动引擎, 请稍候...")
    engine = LocalRAGEngine()

    print("\n===系统就绪(输入quit退出)===")
    while True:
        query = input("\nUser:")
        if query.lower() in ['quit','exit']:
            break

        print("AI 思考中...")
        try:
            response = engine.chat(query)
            ans = response['answer']
        
            if "回答:" in ans:
                ans = ans.split("回答:")[-1].strip()
            
            print(f"AI:{ans}")
            print(f"参考文件:{list(set(response['sources']))}")
        
        except Exception as e:
            print(f"发生错误: {e}")

if __name__=="__main__":
    main()