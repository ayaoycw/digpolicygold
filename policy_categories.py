"""
业务政策分类与搜索模板
=========================
定义四层业务分类体系，为 AI orchestrator 提供结构化参考。

四层：
    基础层 — 注册落户、税收优惠、资质认定等基础政策
    发展层 — 扩产升级、技术创新、融资扶持等发展阶段政策
    人才层 — 人才引进、培训补贴、安居保障等人才政策
    荣誉层 — 奖项评定、示范认定、品牌扶持等荣誉政策
"""

from typing import List, Dict, Any


# ─────────────────────────────────────────────
# 四层业务分类定义
# ─────────────────────────────────────────────

BUSINESS_LAYERS = {
    "基础层": {
        "description": "企业落户、注册、税收优惠、资质认定等基础扶持政策",
        "keywords": ["注册", "落户", "税收优惠", "资质认定", "营业执照", "开办补贴", "租金减免", "园区入驻"],
        "search_template": "{region} {industry} 企业注册落户 税收优惠政策 {year}",
        "prompt_template": (
            "请查找 {region} 面向 {industry} 行业的基础扶持政策，"
            "包括企业注册落户优惠、税收减免、资质认定补贴、园区入驻奖励等。"
        ),
    },
    "发展层": {
        "description": "扩产升级、技术创新、研发投入、融资扶持等发展阶段政策",
        "keywords": ["技术改造", "研发补贴", "融资担保", "上市扶持", "产业升级", "数字化转型", "技术中心", "创新平台"],
        "search_template": "{region} {industry} 研发补贴 技术创新 产业扶持政策 {year}",
        "prompt_template": (
            "请查找 {region} 面向 {industry} 行业的发展阶段政策，"
            "包括研发投入补贴、技术改造资金、融资担保扶持、上市奖励、产业升级项目等。"
        ),
    },
    "人才层": {
        "description": "人才引进、培训补贴、安居保障、团队奖励等人才政策",
        "keywords": ["人才引进", "培训补贴", "安居补贴", "住房保障", "博士后", "创业团队", "技术人才", "高层次人才"],
        "search_template": "{region} {industry} 人才引进 补贴政策 {year}",
        "prompt_template": (
            "请查找 {region} 面向 {industry} 行业相关的人才政策，"
            "包括高层次人才引进奖励、培训补贴、安居保障、创业团队扶持等。"
        ),
    },
    "荣誉层": {
        "description": "奖项评定、示范企业认定、品牌扶持、标杆项目等荣誉政策",
        "keywords": ["示范企业", "品牌认定", "专精特新", "独角兽", "领军企业", "科技小巨人", "创新标杆"],
        "search_template": "{region} {industry} 示范企业 专精特新 认定奖励 {year}",
        "prompt_template": (
            "请查找 {region} 面向 {industry} 行业的荣誉认定政策，"
            "包括专精特新、示范企业认定奖励、品牌扶持、科技小巨人等荣誉项目。"
        ),
    },
}


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def get_layers_reference() -> str:
    """
    获取四层分类参考文本，供 AI prompt 使用。
    """
    lines = ["政策业务分类（四层体系）："]
    for layer, info in BUSINESS_LAYERS.items():
        keywords_str = "、".join(info["keywords"][:5])
        lines.append(f"  - {layer}：{info['description']}（关键词：{keywords_str}）")
    return "\n".join(lines)


def get_search_tasks(
    company_name: str,
    industry: str = "",
    region: str = "",
    year: str = "2025",
) -> List[Dict[str, Any]]:
    """
    根据企业信息，为每一层生成搜索任务。

    Returns:
        [{"layer": "基础层", "search_term": "...", "prompt": "..."}, ...]
    """
    tasks = []
    for layer, info in BUSINESS_LAYERS.items():
        task = {
            "layer": layer,
            "search_term": info["search_template"].format(
                region=region or "全国",
                industry=industry or company_name,
                year=year,
            ),
            "prompt": info["prompt_template"].format(
                region=region or "全国",
                industry=industry or company_name,
            ),
        }
        tasks.append(task)
    return tasks


def get_layer_names() -> List[str]:
    """返回所有层的名称列表"""
    return list(BUSINESS_LAYERS.keys())
