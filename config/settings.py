import os
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str, default: str = None) -> str:
    """统一获取密钥：优先 Streamlit Secrets，回退到 .env 环境变量"""
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val:
            return val
    except Exception:
        pass
    return os.getenv(key, default)


from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults

# 大语言模型实例
llm = ChatOpenAI(
    model=_get_secret("LLM_MODEL", "deepseek-chat"),
    api_key=_get_secret("LLM_API_KEY"),
    base_url=_get_secret("LLM_BASE_URL", "https://api.deepseek.com"),
    temperature=0.2
)

# 中文嵌入模型实例 - 懒加载
_embeddings_instance = None


def get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is not None:
        return _embeddings_instance

    model_name = _get_secret("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    use_real = _get_secret("USE_REAL_EMBEDDINGS", "").lower() in ("1", "true", "yes")

    if use_real:
        print("[INFO] 正在加载嵌入模型...")
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
        from langchain_community.embeddings import FakeEmbeddings
        _embeddings_instance = FakeEmbeddings(size=384)

    return _embeddings_instance


# 联网搜索工具
try:
    from langchain_tavily import TavilySearch
    search_tool = TavilySearch(max_results=3)
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults
    search_tool = TavilySearchResults(max_results=3)
