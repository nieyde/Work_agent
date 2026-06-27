from typing import List, Union
from pydantic import BaseModel, Field, field_validator
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from config.settings import llm


# ==================== 结构化数据模型 ====================
class UserProfile(BaseModel):
    """用户画像结构化模型"""
    name: str = Field(description="姓名")
    education: str = Field(description="最高学历、学校与专业")
    work_years: Union[float, str] = Field(description="工作年限，应届生为0，未知填'未提及'")
    core_skills: List[str] = Field(description="核心硬技能清单，如Python、STM32、Java")
    project_experience: List[str] = Field(description="核心项目经历，每条一句话概括")
    work_experience: List[str] = Field(description="过往工作经历，每条一句话概括")
    target_position: str = Field(description="目标求职岗位")
    expected_salary: str = Field(description="期望薪资范围")
    self_assessment: str = Field(description="自我评价与求职优势")

    @field_validator("work_years", mode="before")
    @classmethod
    def coerce_work_years(cls, v):
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return 0.0
        return 0.0


class JobDescription(BaseModel):
    """岗位JD结构化模型"""
    position_name: str = Field(description="岗位名称")
    department: str = Field(description="所属部门")
    required_education: str = Field(description="学历要求")
    required_years: str = Field(description="工作年限要求")
    hard_skills: List[str] = Field(description="硬性技能要求清单")
    soft_skills: List[str] = Field(description="软素质要求清单")
    job_duties: List[str] = Field(description="主要工作职责")
    salary_range: str = Field(description="薪资范围，未标注则填'未说明'")


# ==================== 解析实现 ====================
def parse_resume(resume_text: str) -> UserProfile:
    """解析简历文本，返回结构化用户画像"""
    parser = PydanticOutputParser(pydantic_object=UserProfile)

    prompt = PromptTemplate(
        template="""
        请从以下简历文本中提取关键信息，输出严格的结构化JSON。
        若某项信息缺失，填写"未提及"，禁止编造内容。

        {format_instructions}

        【简历文本】
        {resume_text}
        """,
        input_variables=["resume_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | llm | parser
    return chain.invoke({"resume_text": resume_text})


def parse_jd(jd_text: str) -> JobDescription:
    """解析JD文本，返回结构化岗位要求"""
    parser = PydanticOutputParser(pydantic_object=JobDescription)

    prompt = PromptTemplate(
        template="""
        请从以下岗位招聘描述中提取要求，输出严格的结构化JSON。
        若某项信息缺失，填写"未说明"，禁止编造内容。

        {format_instructions}

        【岗位描述】
        {jd_text}
        """,
        input_variables=["jd_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | llm | parser
    return chain.invoke({"jd_text": jd_text})
