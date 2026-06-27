from typing import Annotated, List, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # 对话消息历史，add_messages 实现消息追加
    messages: Annotated[list, add_messages]
    # 用户画像（解析简历后填充）
    user_profile: dict
    # 目标岗位 JD 信息
    target_jd: dict
    # 匹配分析结果
    match_report: str
    # 求职规划结果
    plan_result: str
    # 当前思考阶段
    current_stage: str
