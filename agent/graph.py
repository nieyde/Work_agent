import json
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from config.settings import llm
from agent.tools import tools
from memory.long_term import save_profile as memory_save, get_profile as memory_get, load_profiles

# 系统提示词，定义Agent身份与行为规范
SYSTEM_PROMPT = """
你是一位资深、专业的求职规划智能助手，你的职责是为用户提供全流程求职规划服务。

你的工作流程：
1. 先充分理解用户的需求和当前求职阶段
2. 判断是否需要调用工具获取信息或执行分析
3. 基于工具返回的结果，给出专业、结构化、可落地的回复
4. 主动引导用户补充必要信息（如简历、目标岗位）

行为准则：
- 严谨务实，不做空泛建议，所有结论尽量有依据
- 主动推进求职流程，不要被动等待用户提问
- 涉及匹配、规划类任务，必须调用对应工具完成，禁止仅凭记忆回答
- 若信息不足，明确告知用户需要提供什么内容
- 保持专业、客观、鼓励的语气

长期记忆能力：
- 你有 remember_profile、recall_profile、list_profiles 三个记忆工具
- 当用户简历分析完成后，主动调用 remember_profile 保存画像，方便下次访问
- 如果用户之前来过，系统会在首次对话时自动加载历史档案
- 你可以用 recall_profile 查特定用户的档案，用 list_profiles 看所有已存用户
"""


def _convert_to_lc_messages(chat_history: list) -> list:
    """将混合dict和LangChain消息对象的列表统一转为LangChain消息对象
    
    过滤掉 ToolMessage 和带 tool_calls 的 AIMessage（跨轮次校验冲突）
    """
    result = []
    for msg in chat_history:
        if isinstance(msg, ToolMessage):
            continue
        if isinstance(msg, AIMessage) and msg.tool_calls:
            continue
        if isinstance(msg, (HumanMessage, AIMessage)):
            result.append(msg)
        elif isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "assistant":
                result.append(AIMessage(content=content))
            else:
                result.append(HumanMessage(content=content))
        else:
            result.append(HumanMessage(content=str(msg)))
    return result


def _extract_profile(messages: list) -> dict | None:
    """从工具结果中提取用户画像（用于持久化存储）"""
    for msg in reversed(messages):
        if hasattr(msg, "name") and msg.name == "resume_analysis":
            try:
                content = msg.content
                # 支持全角和半角冒号
                for sep in ("：\n", ":\n"):
                    if sep in content:
                        json_str = content.split(sep, 1)[1]
                        return json.loads(json_str)
                # 兜底：尝试整体解析
                return json.loads(content)
            except Exception:
                pass
    return None


def _auto_save_profile(messages: list):
    """对话结束后自动保存用户画像"""
    profile = _extract_profile(messages)
    if not profile:
        return

    name = profile.get("name", "")
    if not name or name == "未提及":
        return

    # 保存到长期记忆
    memory_save(name, profile)
    print(f"[MEMORY] 已保存用户画像: {name}")


def get_saved_user_profiles() -> list[str]:
    """获取所有已存储的用户姓名列表（供 app.py 使用）"""
    return list(load_profiles().keys())


def get_user_context(name: str) -> str | None:
    """
    获取指定用户的历史画像上下文，供首次对话注入。
    返回 None 表示用户不存在。
    """
    profile = memory_get(name)
    if profile:
        return json.dumps(profile, ensure_ascii=False, indent=2)
    return None


# 使用 LangGraph 官方推荐的 create_react_agent 构建 Agent
agent_graph = create_react_agent(
    model=llm.bind_tools(tools),
    tools=tools,
    prompt=SystemMessage(content=SYSTEM_PROMPT.strip()),
)


def chat_with_agent(user_input: str, chat_history: list = None):
    """
    与Agent对话的统一入口
    :param user_input: 用户输入文本
    :param chat_history: 历史消息列表（支持dict和LangChain消息对象混合）
    :return: 回复内容、更新后的消息列表
    """
    if chat_history is None:
        chat_history = []

    normalized = _convert_to_lc_messages(chat_history)
    input_messages = normalized + [HumanMessage(content=user_input)]

    result = agent_graph.invoke({"messages": input_messages})
    messages = result["messages"]

    # 自动保存用户画像到长期记忆
    _auto_save_profile(messages)

    final_message = messages[-1].content
    return final_message, messages
