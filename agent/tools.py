from langchain_core.tools import tool
from core.rag import knowledge_search
from core.parser import parse_resume, parse_jd
from core.matcher import analyze_match
from core.planner import generate_plan
from config.settings import search_tool
from memory.long_term import save_profile as memory_save, get_profile as memory_get, load_profiles


@tool
def search_knowledge(query: str) -> str:
    """
    检索求职专业知识库，回答简历、面试、职业规划类专业问题。
    当用户询问求职方法、技巧、规范时使用此工具。
    参数: query: 用户的问题
    """
    return knowledge_search(query)


@tool
def resume_analysis(resume_text: str) -> str:
    """
    解析用户简历文本，提取结构化用户画像。
    当用户提供简历内容、需要分析个人背景时使用此工具。
    参数: resume_text: 完整的简历文本内容
    """
    profile = parse_resume(resume_text)
    return f"用户画像解析完成：\n{profile.model_dump_json(indent=2)}"


@tool
def jd_analysis(jd_text: str) -> str:
    """
    解析目标岗位JD，提取结构化岗位要求。
    当用户提供岗位描述、需要分析岗位要求时使用此工具。
    参数: jd_text: 完整的岗位招聘描述文本
    """
    jd = parse_jd(jd_text)
    return f"岗位要求解析完成：\n{jd.model_dump_json(indent=2)}"


@tool
def match_analysis(profile_json: str, jd_json: str) -> str:
    """
    对比用户画像与岗位要求，生成人岗匹配度报告与差距分析。
    必须先完成简历解析和JD解析后才能调用此工具。
    参数:
        profile_json: 用户画像的JSON字符串
        jd_json: 岗位要求的JSON字符串
    """
    from core.parser import UserProfile, JobDescription
    import json
    profile = UserProfile(**json.loads(profile_json))
    jd = JobDescription(**json.loads(jd_json))
    return analyze_match(profile, jd)


@tool
def plan_generation(profile_json: str, target_position: str, gap_analysis: str, timeline: str = "3个月") -> str:
    """
    生成个性化分阶段求职规划方案。
    必须先完成差距分析后才能调用此工具。
    参数:
        profile_json: 用户画像的JSON字符串
        target_position: 目标岗位名称
        gap_analysis: 人岗差距分析报告
        timeline: 规划周期，默认3个月
    """
    from core.parser import UserProfile
    import json
    profile = UserProfile(**json.loads(profile_json))
    return generate_plan(profile, target_position, gap_analysis, timeline)


@tool
def internet_search(query: str) -> str:
    """
    联网搜索最新岗位信息、薪资行情、行业趋势、企业招聘动态。
    当需要实时数据、知识库没有的最新信息时使用此工具。
    参数: query: 搜索关键词
    """
    raw = search_tool.invoke(query)
    # 兼容 TavilySearch（返回dict）和 TavilySearchResults（返回list）
    if isinstance(raw, dict):
        results = raw.get("results", [])
    elif isinstance(raw, list):
        results = raw
    else:
        return f"搜索返回了意外格式：{raw}"
    return "\n\n".join([f"来源{i+1}: {res['content']}" for i, res in enumerate(results)])


# ==================== 长期记忆工具 ====================
@tool
def remember_profile(name: str, profile_data: str) -> str:
    """
    保存用户画像到长期记忆，下次对话可以加载。
    当用户简历解析完成、用户画像已生成后，调用此工具保存。
    参数:
        name: 用户姓名（作为记忆索引的key）
        profile_data: 用户画像的完整JSON字符串
    """
    import json
    try:
        profile_dict = json.loads(profile_data)
    except json.JSONDecodeError:
        profile_dict = {"raw": profile_data}
    memory_save(name, profile_dict)
    return f"✅ 用户画像已保存！下次访问时系统会自动加载 {name} 的职业档案。"


@tool
def recall_profile(name: str) -> str:
    """
    从长期记忆中加载用户的职业档案。
    当用户提到之前来过、或者系统启动时自动检查存在哪些用户档案时使用。
    参数:
        name: 用户姓名
    """
    profile = memory_get(name)
    if profile:
        return f"📋 {name} 的职业档案：\n{profile}"
    return f"❌ 未找到 {name} 的档案记录。用户需要先上传简历并完成分析。"


@tool
def list_profiles() -> str:
    """
    列出所有已存储的用户档案名称。
    用于查看系统中目前保存了哪些用户的求职资料。
    参数: 无
    """
    profiles = load_profiles()
    if not profiles:
        return "📭 系统中暂无已存储的用户档案。请先上传简历并完成分析。"
    result = "📋 已存储的用户档案：\n"
    for name, data in profiles.items():
        exp_salary = data.get("expected_salary", "未知")
        target = data.get("target_position", "未设定")
        result += f"  - {name} | 目标: {target} | 期望薪资: {exp_salary}\n"
    return result


# 工具列表
tools = [
    search_knowledge,
    resume_analysis,
    jd_analysis,
    match_analysis,
    plan_generation,
    internet_search,
    remember_profile,
    recall_profile,
    list_profiles,
]
