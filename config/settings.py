import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# 大语言模型实例
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0.2  # 求职场景严谨性优先，低温减少幻觉
)

# 中文嵌入模型实例 - 懒加载，避免导入时下载模型
_embeddings_instance = None

def get_embeddings():
    """获取嵌入模型实例（首次调用时初始化）
    
    默认使用轻量级 FakeEmbeddings（无需下载，适合MVP演示）。
    如需高质量语义搜索，可配置真实嵌入模型：
      1. pip install langchain-huggingface sentence-transformers
      2. 设置环境变量 EMBEDDING_MODEL 为模型名称
    """
    global _embeddings_instance
    if _embeddings_instance is not None:
        return _embeddings_instance

    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    use_real = os.getenv("USE_REAL_EMBEDDINGS", "").lower() in ("1", "true", "yes")

    if use_real:
        print("[INFO] 正在加载嵌入模型（首次使用需下载，约 300MB）...")
        try:
            from langchain_huggingface import HuggingFaceEmbeddings as HFEmbeddings
            _embeddings_instance = HFEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        except ImportError:
            _embeddings_instance = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
    else:
        print("[INFO] 使用 FakeEmbeddings（轻量模式，RAG结果为关键词匹配）")
        from langchain_community.embeddings import FakeEmbeddings
        _embeddings_instance = FakeEmbeddings(size=384)
    
    print("[INFO] 嵌入模型初始化完成")
    return _embeddings_instance

# 注意：其他模块请通过 get_embeddings() 函数获取，而非直接引用 embeddings 变量

# 联网搜索工具实例
try:
    from langchain_tavily import TavilySearch
    search_tool = TavilySearch(max_results=3)
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults
    search_tool = TavilySearchResults(max_results=3)
