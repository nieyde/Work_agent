import streamlit as st
from pypdf import PdfReader
import docx
from agent.graph import chat_with_agent, get_saved_user_profiles, get_user_context

st.set_page_config(page_title="求职规划智能体", layout="wide")
st.title("💼 求职规划智能体 JobPlanning Agent")
st.caption("基于 LangGraph 的全流程求职规划助手 · 支持长期记忆")

# ==================== 会话状态初始化 ====================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "resume_shared" not in st.session_state:
    st.session_state.resume_shared = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "profile_loaded" not in st.session_state:
    st.session_state.profile_loaded = False


def _ensure_resume_shared():
    """确保简历已共享给 Agent。"""
    if not st.session_state.resume_text:
        return False
    if st.session_state.resume_shared:
        return True

    _, messages = chat_with_agent(
        f"请先分析我的简历，生成结构化用户画像：\n{st.session_state.resume_text}",
        st.session_state.messages
    )
    st.session_state.messages = messages
    st.session_state.resume_shared = True
    return True


def _load_existing_profile(user_name: str):
    """加载已保存的用户画像并注入到对话中。"""
    profile_json = get_user_context(user_name)
    if not profile_json:
        return False

    # 将画像注入为第一条对话消息
    _, messages = chat_with_agent(
        f"我是 {user_name}，我的职业档案如下，请回顾我的情况：\n{profile_json}",
        st.session_state.messages
    )
    st.session_state.messages = messages
    st.session_state.current_user = user_name
    st.session_state.profile_loaded = True
    return True


# ==================== 侧边栏 ====================
with st.sidebar:
    # ---- 用户档案区 ----
    st.header("👤 我的档案")
    existing_users = get_saved_user_profiles()

    if existing_users:
        st.caption("检测到已存储的用户档案：")
        for uname in existing_users:
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(f"📋 {uname}", key=f"load_{uname}", use_container_width=True):
                    with st.spinner(f"正在加载 {uname} 的档案..."):
                        _load_existing_profile(uname)
                        st.rerun()
            with col2:
                if st.button("🗑", key=f"del_{uname}"):
                    from memory.long_term import delete_profile
                    delete_profile(uname)
                    st.rerun()
        st.divider()
    else:
        st.info("暂无已存储档案")

    # ---- 简历上传区 ----
    st.header("📄 简历上传")
    uploaded_file = st.file_uploader(
        "上传你的简历（PDF/Word）",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            reader = PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() for page in reader.pages])
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            text = uploaded_file.read().decode("utf-8")

        st.session_state.resume_text = text
        st.session_state.resume_shared = False
        st.success("简历解析完成，可在对话中使用")

        if st.button("🔍 一键分析我的简历"):
            with st.spinner("Agent 正在分析你的简历..."):
                reply, messages = chat_with_agent(
                    f"这是我的简历，请帮我分析并生成用户画像：\n{st.session_state.resume_text}",
                    st.session_state.messages
                )
                st.session_state.messages = messages
                st.session_state.resume_shared = True
                st.rerun()

    st.divider()
    st.header("⚙️ 快捷功能")
    if st.button("📅 生成求职规划"):
        if not st.session_state.resume_text and st.session_state.current_user is None:
            st.warning("请先上传简历或加载已有档案")
        else:
            _ensure_resume_shared()
            with st.spinner("Agent 正在生成规划..."):
                reply, messages = chat_with_agent(
                    "基于我的简历，帮我生成一份3个月的求职规划方案",
                    st.session_state.messages
                )
                st.session_state.messages = messages
                st.rerun()

# ==================== 主聊天区 ====================
# 首次进入且无对话但有已存档案 → 显示提示
if not st.session_state.messages and existing_users and not st.session_state.profile_loaded:
    st.info(f"👋 欢迎回来！检测到你是老用户。请在左侧「我的档案」中点击你的名字加载历史资料。")

# 显示历史消息
for msg in st.session_state.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ==================== 用户输入 ====================
if prompt := st.chat_input("请输入你的问题，比如：帮我分析这个岗位的匹配度..."):
    # 自动共享简历
    if st.session_state.resume_text and not st.session_state.resume_shared:
        _ensure_resume_shared()

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent 思考中..."):
            reply, messages = chat_with_agent(prompt, st.session_state.messages)
            st.markdown(reply)
            st.session_state.messages = messages
