import os
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from config.settings import llm, get_embeddings

VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH")
KB_PATH = os.getenv("KNOWLEDGE_BASE_PATH")


def build_vector_store():
    """构建向量库并持久化"""
    loader = DirectoryLoader(
        KB_PATH,
        glob="**/*",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "；", " ", ""]
    )
    splits = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(splits, get_embeddings())
    vectorstore.save_local(VECTOR_STORE_PATH)
    return vectorstore


def get_vector_store():
    """加载本地向量库，不存在则新建"""
    if os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
        return FAISS.load_local(VECTOR_STORE_PATH, get_embeddings(), allow_dangerous_deserialization=True)
    return build_vector_store()


def format_docs(docs):
    """将检索到的文档拼接为文本"""
    return "\n\n---\n\n".join([doc.page_content for doc in docs])


def knowledge_search(query: str) -> str:
    """知识库检索工具入口，供Agent调用"""
    vectorstore = get_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt_template = PromptTemplate(
        template="""你是专业的求职规划顾问，请严格基于下方参考资料回答用户问题。
若参考资料无相关内容，请明确说明"知识库暂无相关内容"，禁止编造信息。
回答需务实、结构化、可落地，避免空泛表述。

【参考资料】
{context}

【用户问题】
{question}

【专业回答】""",
        input_variables=["context", "question"]
    )

    # 构建检索增强链
    rag_chain = (
        RunnableParallel(
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(query)
