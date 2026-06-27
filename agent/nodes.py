from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import llm
from agent.tools import tools

# 绑定工具到LLM
llm_with_tools = llm.bind_tools(tools)

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
"""


def agent_node(state):
    """推理决策节点：大模型思考并决定是否调用工具"""
    # 仅当消息列表中没有 SystemMessage 时才添加（避免循环中重复追加）
    has_system = any(isinstance(m, SystemMessage) for m in state["messages"])
    if has_system:
        messages = state["messages"]
    else:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "current_stage": "thinking" if response.tool_calls else "answering"
    }


# 工具执行节点
tool_node = ToolNode(tools)


def should_continue(state):
    """条件边判断：决定继续调用工具还是生成最终回复"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"


def extract_profile_node(state):
    """辅助节点：从工具结果中提取用户画像存入状态"""
    messages = state["messages"]
    # 遍历消息找到简历解析结果
    for msg in reversed(messages):
        if msg.name == "resume_analysis":
            import json
            try:
                content = msg.content
                json_str = content.split(":\n", 1)[1]
                profile = json.loads(json_str)
                return {"user_profile": profile}
            except:
                pass
    return {}
