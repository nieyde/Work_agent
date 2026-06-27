from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import llm
from core.parser import UserProfile, JobDescription


def analyze_match(profile: UserProfile, jd: JobDescription) -> str:
    """生成人岗匹配分析报告"""
    prompt = PromptTemplate(
        template="""
        你是资深招聘专家，请基于用户画像与岗位要求，生成专业的人岗匹配分析报告。
        报告需包含以下4个部分：
        1. 综合匹配度评分（0-100分）与整体评价
        2. 用户核心优势（与岗位匹配的点）
        3. 存在的差距与不足（分硬技能、经验、学历三类）
        4. 针对性弥补建议（按优先级排序）

        【用户画像】
        {profile}

        【岗位要求】
        {jd}

        【分析报告】
        """,
        input_variables=["profile", "jd"]
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "profile": profile.model_dump_json(),
        "jd": jd.model_dump_json()
    })
