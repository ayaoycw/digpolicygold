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


# ─────────────────────────────────────────────
# 五维度特征分析框架（专家系统）
# ─────────────────────────────────────────────

EXPERT_DIMENSIONS = {
    "空间载体": {
        "id": "spatial",
        "description": "从注册地址推导园区/功能区，匹配园区级专项政策",
        "zone_types": {
            "经开区(ETDZ)": {"keywords": ["经济技术开发区", "经开区", "出口加工区"], "policy_focus": ["重大项目奖励", "亩均效益"]},
            "高新区(HIDZ)": {"keywords": ["高新", "火炬", "软件园"], "policy_focus": ["研发加计扣除", "瞪羚", "独角兽"]},
            "自贸区(FTZ)": {"keywords": ["自贸试验区", "保税区", "围网", "综保区"], "policy_focus": ["跨境金融", "关税免退", "制度创新"]},
            "科创载体": {"keywords": ["大学", "科技大厦", "小镇", "科技园", "产业园"], "policy_focus": ["产学研", "房租减免"]},
        },
    },
    "产业链": {
        "id": "industry_chain",
        "description": "通过经营范围和知识产权推断产业链地位",
        "chain_roles": {
            "链主/总部": {"keywords": ["总部管理", "供应链管理", "投资管理"], "policy_focus": ["总部经济", "一事一议"]},
            "关键环节": {"keywords": ["精密加工", "核心零部件", "关键材料", "清洗设备"], "policy_focus": ["强链补链", "首台套", "首批次"]},
        },
        "ip_tiers": {
            "I类(高权重)": {"types": ["发明专利", "国家标准"], "policy_focus": ["专精特新", "高企认定"]},
            "II类(基础)": {"types": ["实用新型", "软件著作权"], "policy_focus": ["科技型中小企业"]},
        },
    },
    "身份属性": {
        "id": "identity",
        "description": "从股东结构判断企业身份，触发身份专属政策",
        "identity_types": {
            "外资(Foreign)": {"signals": ["境外法人", "境外个人", "香港", "台湾", "外国"], "policy_focus": ["外资研发中心免退税", "跨国公司地区总部"]},
            "国资(SOE)": {"signals": ["国资委", "地方国资", "国有"], "policy_focus": ["合规管理", "重大基建"]},
            "高校(Academic)": {"signals": ["资产管理公司", "大学", "学院", "研究院"], "policy_focus": ["产学研合作", "成果转化奖补"]},
        },
    },
    "人力资源": {
        "id": "hr_dynamics",
        "description": "利用参保人数变动推断用工动态，匹配人力政策",
        "hr_signals": {
            "增员(Growth)": {"condition": "参保人数同比增长>30%", "policy_focus": ["一次性扩岗补助", "人才安家费", "应届生补贴"]},
            "稳员(Stability)": {"condition": "参保人数稳定", "policy_focus": ["稳岗返还", "社保补贴"]},
            "高管画像": {"condition": "主要人员含博士/海归", "policy_focus": ["高层次人才认定", "科研启动金", "人才公寓"]},
        },
    },
    "合规熔断": {
        "id": "compliance",
        "description": "检查企业合规状态，决定是否触发政策绝缘",
        "veto_rules": {
            "一票否决": ["严重违法失信", "经营异常名录（未移出）"],
            "时效分析": "轻微失信已过公示期可恢复；严重失信需1-3年修复期",
            "修复路径": "通过'信用修复'关键词寻找退出机制",
        },
    },
    "税收与财务": {
        "id": "tax_financial",
        "description": "搜索企业可享受的税收优惠和财务激励（加计扣除、所得税减免、增值税即征即退等）",
        "tax_types": {
            "研发加计扣除": {"condition": "所有科技型企业默认触发", "policy_focus": ["研发费用加计扣除", "200%加计扣除"]},
            "企业所得税优惠": {"condition": "高企/自贸区/小微企业", "policy_focus": ["高企所得税15%", "小微企业所得税减免", "重点产业15%税率"]},
            "增值税优惠": {"condition": "软件/集成电路行业", "policy_focus": ["增值税即征即退", "软件产品增值税"]},
        },
    },
    "人才激励": {
        "id": "talent_incentive",
        "description": "搜索面向员工个人的政府奖励（重点产业人才奖励、落户、职称评定），区别于企业层面的人力资源补贴",
        "incentive_types": {
            "重点产业人才专项奖励": {"condition": "企业在集成电路/AI/通信等重点产业", "policy_focus": ["重点产业人才专项奖励", "人才个人奖励150万"]},
            "高层次人才落户": {"condition": "员工含博士/海归/高级职称", "policy_focus": ["高层次人才落户", "居转户绿色通道"]},
            "科技成果转化奖励": {"condition": "有技术成果转化", "policy_focus": ["科技成果转化人才奖励"]},
        },
    },
}


def get_dimensions_reference() -> str:
    """
    获取七维度分析框架参考文本，供 AI prompt 使用。
    """
    lines = ["企业特征分析维度（七维度框架）："]
    for dim_name, dim_info in EXPERT_DIMENSIONS.items():
        lines.append(f"  - {dim_name}：{dim_info['description']}")
    return "\n".join(lines)


def get_dimension_names() -> List[str]:
    """返回所有维度名称列表"""
    return list(EXPERT_DIMENSIONS.keys())
