from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import llm
from core.parser import UserProfile


def generate_plan(profile: UserProfile, target_position: str, gap_analysis: str, timeline: str = "3个月") -> str:
    """生成个性化求职路径规划"""
    prompt = PromptTemplate(
        template="""
        你是资深职业规划师，请基于用户现状、目标岗位和差距分析，生成{timeline}的分阶段求职执行规划。
        要求：
        1. 按周拆分阶段，明确每个阶段的核心目标
        2. 每个阶段给出具体可执行的任务清单，可量化、可验证
        3. 覆盖技能提升、项目补充、简历优化、投递策略、面试准备5个维度
        4. 给出每周时间分配建议与验收标准

        【用户现状】
        {profile}

        【目标岗位】
        {target_position}

        【差距分析】
        {gap_analysis}

        【详细规划方案】
        """,
        input_variables=["profile", "target_position", "gap_analysis", "timeline"]
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "profile": profile.model_dump_json(),
        "target_position": target_position,
        "gap_analysis": gap_analysis,
        "timeline": timeline
    })
