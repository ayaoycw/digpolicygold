"""
Orchestrator — AI 智能调度
==========================
根据企查查企业信息，AI 思考后拆分搜索任务，并动态决定是否调用 browse use。

流程：
    1. 接收企查查企业信息（名称、行业、地区、标签等）
    2. AI 分析企业特征 → 生成若干 web search 任务
    3. 并行执行 web search
    4. AI 评估搜索结果 → 决定哪些需要 browse use 深度抓取
    5. 执行 browse use（可选）
    6. 合并 + 去重 → 返回最终结果

用法：
    from orchestrator import Orchestrator

    orch = Orchestrator()
    # 仅 AI 拆分（不执行搜索，用于调试）
    plan = orch.plan(company_info)
    # 完整流程
    result = await orch.run(company_info)
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.web_search")

from models import PolicyItem, WorkerResult
from policy_categories import get_layers_reference, get_dimensions_reference

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 加载专家 Prompt 文件
# ─────────────────────────────────────────────

_PROMPT_DIR = Path(__file__).parent / "prompts"

def _load_prompt(filename: str) -> str:
    """从 prompts/ 目录加载 markdown prompt 文件"""
    filepath = _PROMPT_DIR / filename
    if filepath.exists():
        return filepath.read_text(encoding="utf-8")
    logger.warning(f"Prompt 文件不存在: {filepath}")
    return ""


# ─────────────────────────────────────────────
# AI 思考 Prompt
# ─────────────────────────────────────────────

def _build_plan_system_prompt() -> str:
    """构建 PLAN 阶段的 system prompt，融合专家框架 + 四层分类 + 输出格式"""
    expert_knowledge = _load_prompt("expert_system_prompt.md")
    layers_ref = get_layers_reference()
    dimensions_ref = get_dimensions_reference()

    return (
        "你是一个政策搜索调度专家。你的任务是根据企业的工商数据，进行深度特征工程，并生成精准的搜索任务列表。\n\n"
        "# 专家认知框架\n\n"
        f"{expert_knowledge}\n\n"
        "# 政策分类参考\n\n"
        f"{layers_ref}\n\n"
        f"{dimensions_ref}\n\n"
        "# 执行规则\n\n"
        "【核心原则】\n"
        "1. 在\"微观颗粒度特征映射\"的基础上，必须保留\"地区+行业+补贴\"的基础搜索模式\n"
        "2. 先对企业数据进行【特征逆向工程】，再生成搜索任务\n"
        "3. 输出的 tasks 中，每个搜索词必须具体、可直接在 Bing/百度 搜索\n"
        "4. 空间载体搜索词必须包含 [园区名称] + 管委会/专项资金，而非仅搜市级政策\n"
        "5. 合规熔断检查：如果企业有严重失信，设置 compliance_veto.passed = false\n\n"
        "【搜索词质量要求 — 极其重要】\n"
        "- 每个维度至少 1-3 个搜索词\n"
        "- 搜索词不要超过 20 个字\n"
        "- ⚠️ 所有搜索词必须包含企业所在地区（如\"上海\"\"浦东新区\"），严禁生成不带地区的搜索词\n"
        "- ⚠️ 产业链维度必须保留至少一个\"地区+行业+补贴/扶持\"的基础搜索词（如\"浦东新区 光通信 产业扶持政策\"），这是搜索引擎最擅长的模式\n"
        "- ⚠️ 产业链术语（强链补链、首台套等）作为额外补充搜索，不替代基础模式，且必须带地区前缀\n"
        "- 优先搜索园区级 > 区级 > 市级政策\n"
        "- 外资企业额外搜索\"外资研发中心\"相关政策\n"
        "- 高校背景企业额外搜索\"产学研合作\"相关政策\n\n"
        "【输出格式 — 严格 JSON】\n"
        "{{\n"
        "  \"feature_engineering\": {{\n"
        "    \"spatial\": \"空间载体分析结果\",\n"
        "    \"industry_chain\": \"产业链地位分析\",\n"
        "    \"identity\": \"身份属性分析\",\n"
        "    \"hr_dynamics\": \"人力资源动态分析\",\n"
        "    \"compliance\": \"合规状态分析\",\n"
        "    \"tax_financial\": \"税收与财务优惠分析\",\n"
        "    \"talent_incentive\": \"人才激励政策分析\"\n"
        "  }},\n"
        "  \"gap_analysis\": {{\n"
        "    \"money\": \"补贴潜力评估\",\n"
        "    \"qualification\": \"资质潜力评估\",\n"
        "    \"talent\": \"人才政策潜力\",\n"
        "    \"compliance\": \"合规风险评估\"\n"
        "  }},\n"
        "  \"analysis\": \"综合分析（2-3句话）\",\n"
        "  \"tasks\": [\n"
        "    {{\n"
        "      \"dimension\": \"空间载体|产业链|身份属性|人力资源|合规|税收与财务|人才激励\",\n"
        "      \"layer\": \"基础层|发展层|人才层|荣誉层\",\n"
        "      \"search_term\": \"具体搜索关键词\",\n"
        "      \"priority\": \"high|medium|low\",\n"
        "      \"reason\": \"搜索意图说明\",\n"
        "      \"focus_hints\": \"给搜索模型的重点关注指引\"\n"
        "    }}\n"
        "  ],\n"
        "  \"compliance_veto\": {{\n"
        "    \"passed\": true,\n"
        "    \"risk_level\": \"none|low|medium|high|blocked\",\n"
        "    \"detail\": \"合规判断说明\"\n"
        "  }}\n"
        "}}\n"
    )

EVALUATE_SYSTEM_PROMPT = """你是一个政策搜索质量评估专家。根据 web search 搜索结果，判断哪些政策条目需要用浏览器深度抓取。

【需要深度抓取的情况】
1. 摘要不完整，只有"..."或过于简短（少于20字）
2. URL 指向 PDF 文件（.pdf 结尾）
3. 缺少关键信息（如扶持金额、申报条件）
4. 来自政府官网(.gov.cn)但摘要模糊，可能包含更详细的政策原文
5. 标题看起来很相关但没有具体内容

【不需要深度抓取的情况】
1. 摘要已经包含完整的政策要点和金额
2. 来自新闻聚合站，内容可能只是转载
3. URL 已失效或不可访问的迹象

输出严格 JSON：
{{
  "evaluation": "整体评价（1句话）",
  "browse_targets": [
    {{
      "title": "政策标题",
      "url": "需要深度抓取的URL",
      "reason": "为什么需要深度抓取"
    }}
  ],
  "skip_reasons": ["跳过项1的原因", "跳过项2的原因"]
}}
"""

# ── 打分排序 Prompt ──

def _build_scoring_system_prompt() -> str:
    """动态生成打分 system prompt，注入当前日期，使用5维度评分体系"""
    today = datetime.now().strftime("%Y-%m-%d")
    current_year = datetime.now().year
    return (
        f"你是一个政策匹配评分专家。当前日期为 {today}。\n"
        "你需要用5个维度为每条政策打分，然后加权计算综合分。\n\n"

        "【重要原则】\n"
        "- 每条政策都必须认真评分，不要给0分（除非与企业完全无关，如外省政策）\n"
        "- 可得性低不是淘汰理由，而是改进建议方向（企业可以为之努力）\n"
        "- 时效性是硬标准：过期政策必须在紧迫性维度体现，但其他维度正常评分\n"
        "- 搜索阶段重覆盖率，评分用于排序而非筛除\n\n"

        "【5维度评分体系（每个维度 0-100 分）】\n\n"

        "1. 💰 金额价值 score_amount（权重30%）— 企业能拿到多少钱\n"
        "   100分: >500万（S级）  80分: 100-500万（A级）  60分: 20-100万（B级）\n"
        "   40分: 5-20万（C级）  20分: <5万（D级）\n"
        "   门槛型政策按撬动价值评估。税收优惠折算实际省税金额。\n\n"

        "2. 🎯 独占性 score_exclusivity（权重25%）— 竞争对手多不多\n"
        "   100分: 定制型(<50家)  80分: 行业型(<200家)  60分: 园区型\n"
        "   40分: 地区型  20分: 普惠型\n\n"

        "3. ✅ 可得性 score_feasibility（权重10%）— 企业当前能否满足条件\n"
        "   ⚠️ 可得性低≠不重要！低可得性说明企业需要为之努力（如先拿高企认定），是改进建议的好方向。\n"
        "   100分: 全满足  80分: 缺1项非关键  60分: 缺1-2项可短期补齐\n"
        "   40分: 缺关键条件需6月+  20分: 基本不满足（但仍应展示给用户）\n\n"

        "4. ⏰ 紧迫性 score_urgency（权重25%）— 时效性，是否还能申报\n"
        f"   ⚠️ 这是最重要的维度。当前日期 {today}，严格判断！\n"
        f"   100分: 申报截止<30天\n"
        "   80分:  截止30-90天\n"
        "   60分:  半年内或常年可申\n"
        "   40分:  预计下批次开放（如年度政策等新一轮）\n"
        "   20分:  有效期已过，但预计有接续政策（如十四五→十五五），仍有参考价值\n"
        "   5分:   有效期已过，无接续迹象\n\n"

        "5. 🔄 持续性 score_sustainability（权重10%）— 可否反复获得\n"
        "   100分: 每年可申  80分: 周期性  60分: 一次性+门槛\n"
        "   40分: 纯一次性  20分: 一次性且小额\n\n"

        f"【时效性判断规则 — 当前 {today}】\n"
        f"- 有效期标注'至2025-12-31'且今天是{today} → 已过期 → score_urgency≤20\n"
        f"- 年度申报通知已截止 → score_urgency=5\n"
        f"- 十四五框架政策(2021-2025) → score_urgency=20（可能有十五五接续）\n"
        f"- 发布超3年无'长期有效' → score_urgency最高40\n"
        f"- 2026年新发布且在申报期 → score_urgency≥80\n"
        f"- 与企业行业/地区完全无关 → 所有维度≤10\n\n"

        "【有效期判断】\n"
        "- 标题/摘要有年份范围→提取  - 已有validity→直接使用\n"
        "- 长期政策→'长期有效'  - 无法判断→'请查原文确认'\n\n"

        "【金额提取与分级】\n"
        "- amount_level: S(>500万)/A(100-500万)/B(20-100万)/C(5-20万)/D(<5万)/?（未知）\n\n"

        "【综合分】relevance = amount×0.3 + exclusivity×0.25 + urgency×0.25 + feasibility×0.1 + sustainability×0.1\n"
        "（四舍五入取整。最低分5分，不要给0分，除非完全无关。）\n\n"

        "输出严格 JSON：\n"
        "{{\n"
        "  \"scored_policies\": [\n"
        "    {{\n"
        "      \"index\": 1,\n"
        "      \"score_amount\": 80,\n"
        "      \"score_exclusivity\": 60,\n"
        "      \"score_feasibility\": 70,\n"
        "      \"score_urgency\": 80,\n"
        "      \"score_sustainability\": 60,\n"
        "      \"relevance\": 72,\n"
        "      \"validity\": \"2026-12-31\",\n"
        "      \"amount\": \"最高500万\",\n"
        "      \"amount_level\": \"A\",\n"
        "      \"reason\": \"评分理由（含独占性和可得性判断）\"\n"
        "    }}\n"
        "  ]\n"
        "}}\n"
    )

# ── 回路评估 Prompt ──

def _build_round_review_system_prompt() -> str:
    """构建回路评估 system prompt，按5维度+4层双重评估"""
    expert_knowledge = _load_prompt("expert_system_prompt.md")
    layers_ref = get_layers_reference()
    dimensions_ref = get_dimensions_reference()

    return (
        "你是一个政策搜索质量评审专家。你刚完成了一轮搜索，现在需要判断结果质量。\n\n"
        "# 专家认知框架\n\n"
        f"{expert_knowledge}\n\n"
        "# 参考分类\n\n"
        f"{layers_ref}\n\n"
        f"{dimensions_ref}\n\n"
        "【你的任务】\n"
        "根据企业特征工程结果和已搜到的政策，同时从两个角度评估覆盖度：\n\n"
        "A. 维度覆盖（7维度）：\n"
        "1. 空间载体 — 是否搜到了园区级/功能区级政策？\n"
        "2. 产业链 — 是否覆盖了企业在产业链中的关键环节政策？\n"
        "3. 身份属性 — 外资/国资/高校等身份专属政策是否已覆盖？\n"
        "4. 人力资源 — 增员/稳员/人才认定相关政策是否已覆盖？\n"
        "5. 合规 — 如有合规风险，是否搜到了信用修复路径？\n"
        "6. 税收与财务 — 是否搜到了研发加计扣除、企业所得税优惠、增值税减免等税收政策？\n"
        "7. 人才激励 — 是否搜到了面向员工个人的政府奖励（重点产业人才奖励、落户、职称评定）？\n\n"
        "B. 业务层覆盖（4层）：基础层/发展层/人才层/荣誉层 各有多少有效结果？\n\n"
        "【判断标准】\n"
        "- 每个相关维度至少有 1-2 条有效政策（含 URL 和扶持内容）算\"足够\"\n"
        "- 某维度 0 条结果，或全部摘要空白 → \"不足\"\n"
        "- 搜索词太泛（结果不相关）或太窄（0 结果）→ 需要调整\n"
        "- 补充搜索词应遵循专家框架的搜索构建规则（园区级优先）\n\n"
        "输出严格 JSON：\n"
        "{{\n"
        "  \"overall_quality\": \"good|fair|poor\",\n"
        "  \"quality_reason\": \"整体质量判断原因（1-2句话）\",\n"
        "  \"dimension_coverage\": {{\n"
        "    \"空间载体\": {{\"status\": \"sufficient|insufficient|missing|not_applicable\", \"count\": 0, \"note\": \"...\"}},\n"
        "    \"产业链\": {{\"status\": \"sufficient|insufficient|missing|not_applicable\", \"count\": 0, \"note\": \"...\"}},\n"
        "    \"身份属性\": {{\"status\": \"sufficient|insufficient|missing|not_applicable\", \"count\": 0, \"note\": \"...\"}},\n"
        "    \"人力资源\": {{\"status\": \"sufficient|insufficient|missing|not_applicable\", \"count\": 0, \"note\": \"...\"}},\n"
        "    \"合规\": {{\"status\": \"sufficient|insufficient|missing|not_applicable\", \"count\": 0, \"note\": \"...\"}},\n"
        "    \"税收与财务\": {{\"status\": \"sufficient|insufficient|missing|not_applicable\", \"count\": 0, \"note\": \"...\"}},\n"
        "    \"人才激励\": {{\"status\": \"sufficient|insufficient|missing|not_applicable\", \"count\": 0, \"note\": \"...\"}}\n"
        "  }},\n"
        "  \"layer_coverage\": {{\n"
        "    \"基础层\": {{\"status\": \"sufficient|insufficient|missing\", \"count\": 0, \"note\": \"...\"}},\n"
        "    \"发展层\": {{\"status\": \"sufficient|insufficient|missing\", \"count\": 0, \"note\": \"...\"}},\n"
        "    \"人才层\": {{\"status\": \"sufficient|insufficient|missing\", \"count\": 0, \"note\": \"...\"}},\n"
        "    \"荣誉层\": {{\"status\": \"sufficient|insufficient|missing\", \"count\": 0, \"note\": \"...\"}}\n"
        "  }},\n"
        "  \"timeliness\": {{\n"
        "    \"status\": \"good|poor\",\n"
        "    \"current_year_count\": 0,\n"
        "    \"outdated_count\": 0,\n"
        "    \"note\": \"时效性说明（当年政策数量、过期政策比例等）\"\n"
        "  }},\n"
        "  \"need_more_search\": true,\n"
        "  \"retry_tasks\": [\n"
        "    {{\n"
        "      \"dimension\": \"需要补充的维度\",\n"
        "      \"layer\": \"对应的业务层\",\n"
        "      \"search_term\": \"改进后的搜索词\",\n"
        "      \"reason\": \"为什么需要重新搜索\"\n"
        "    }}\n"
        "  ]\n"
        "}}\n"
    )


# ─────────────────────────────────────────────
# Orchestrator 主类
# ─────────────────────────────────────────────

class Orchestrator:
    """
    AI 智能调度器（带评估反馈回路）

    Args:
        on_log:       日志回调（可选，用于 SSE 推送）
        time_budget:  总时间预算（秒），超时后不再启动新搜索轮次
        max_rounds:   最大搜索轮次（含首轮）
        request_delay: 每次 web search 请求间隔（秒），避免 429
    """

    def __init__(
        self,
        on_log: Optional[Callable[[str], None]] = None,
        time_budget: float = 360.0,
        max_rounds: int = 3,
        request_delay: float = 2.0,
    ):
        self.on_log = on_log or (lambda msg: logger.info(msg))
        self.time_budget = time_budget
        self.max_rounds = max_rounds
        self.request_delay = request_delay
        self._client = None
        self._start_time: float = 0.0

    def _log(self, msg: str):
        self.on_log(msg)

    def _elapsed(self) -> float:
        """已用时间（秒）"""
        return round(time.time() - self._start_time, 1)

    def _time_remaining(self) -> float:
        """剩余时间（秒）"""
        return max(0, self.time_budget - (time.time() - self._start_time))

    def _is_timeout(self) -> bool:
        """是否已超时"""
        return time.time() - self._start_time >= self.time_budget

    def _ensure_client(self):
        """延迟初始化 Azure OpenAI 客户端"""
        if self._client is not None:
            return
        from openai import AzureOpenAI

        # 复用 web_search_worker 的配置
        project_endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "")
        from urllib.parse import urlparse
        parsed = urlparse(project_endpoint)
        endpoint = f"{parsed.scheme}://{parsed.netloc}" if project_endpoint else ""

        self._client = AzureOpenAI(
            api_key=os.environ.get("AZURE_AI_API_KEY", ""),
            api_version="2025-04-01-preview",
            azure_endpoint=endpoint,
        )

    def _ai_call(self, system_prompt: str, user_content: str) -> dict:
        """
        调用 AI（GPT-4o）进行思考，返回 JSON dict。
        """
        self._ensure_client()
        model = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o")

        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        text = response.choices[0].message.content or "{}"
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试提取 JSON 块
            import re
            m = re.search(r'\{[\s\S]*\}', text)
            if m:
                return json.loads(m.group())
            return {"error": "AI 返回了非 JSON 内容", "raw": text}

    # ─────────────────────────────────────
    # Step 1: AI 拆分任务
    # ─────────────────────────────────────

    # ─────────────────────────────────────
    # 构建 User Prompt
    # ─────────────────────────────────────

    @staticmethod
    def _build_user_content(company_info: Dict[str, Any]) -> str:
        """
        从 company_info 构建结构化的 user prompt。
        支持丰富字段（方案 A），缺失字段优雅降级。

        扩展字段清单：
            name, industry, region, tags,
            address,              # 注册地址全文（空间载体分析）
            business_scope,       # 经营范围（产业链分析）
            registered_capital,   # 注册资本
            shareholders,         # [{"name": ..., "type": ..., "ratio": ...}]（身份属性）
            ip,                   # {"invention": 8, "utility": 20, "software": 5}（IP分级）
            headcount_history,    # {"2023": 45, "2024": 82}（HR动态）
            employees,            # 员工规模（简略）
            founded,              # 成立时间
            risk_info,            # 风险信息文本（合规熔断）
        """
        # 加载 user prompt 模板
        template = _load_prompt("expert_user_prompt.md")

        # ── 构建各字段 ──
        name = company_info.get("name", "未知")
        industry = company_info.get("industry", "未知")
        region = company_info.get("region", "未知")
        address = company_info.get("address", "未提供")
        business_scope = company_info.get("business_scope", "未提供")
        registered_capital = company_info.get("registered_capital", "未提供")
        founded = company_info.get("founded", "未提供")
        employees = company_info.get("employees", "未提供")
        tags = ", ".join(company_info.get("tags", [])) or "无"
        risk_info = company_info.get("risk_info", "未提供")

        # IP
        ip_data = company_info.get("ip", {})
        ip_invention = ip_data.get("invention", "未知")
        ip_utility = ip_data.get("utility", "未知")
        ip_software = ip_data.get("software", "未知")

        # 股东
        shareholders = company_info.get("shareholders", [])
        if shareholders:
            sh_lines = []
            for sh in shareholders:
                sh_name = sh.get("name", "?")
                sh_type = sh.get("type", "?")
                sh_ratio = sh.get("ratio", "?")
                sh_lines.append(f"  - {sh_name}（{sh_type}，持股 {sh_ratio}）")
            shareholders_text = "\n".join(sh_lines)
        else:
            shareholders_text = "  未提供"

        # 参保人数历史
        headcount = company_info.get("headcount_history", {})
        if headcount:
            hc_lines = [f"  - {year}年：{count}人" for year, count in sorted(headcount.items())]
            headcount_text = "\n".join(hc_lines)
        else:
            headcount_text = "  未提供"

        # 如果有模板文件，用模板；否则用行内格式
        if template:
            try:
                user_content = template.format(
                    name=name, industry=industry, region=region,
                    address=address, business_scope=business_scope,
                    registered_capital=registered_capital,
                    founded=founded, employees=employees, tags=tags,
                    risk_info=risk_info,
                    ip_invention=ip_invention, ip_utility=ip_utility, ip_software=ip_software,
                    shareholders_text=shareholders_text,
                    headcount_text=headcount_text,
                )
                # 追加 Step 0 补充信息
                extras = []
                actual_addr = company_info.get("actual_address", "")
                if actual_addr:
                    extras.append(f"- ⚠️ 实际办公地址：{actual_addr}（与注册地址不同，需同时搜两个区的政策）")
                core_products = company_info.get("core_products", "")
                if core_products:
                    extras.append(f"- 🔬 核心产品/技术路线：{core_products}")
                certifications = company_info.get("certifications", [])
                if certifications:
                    extras.append(f"- 🏅 已获资质：{', '.join(certifications)}")
                founder_bg = company_info.get("founder_background", "")
                if founder_bg:
                    extras.append(f"- 👤 创始人背景：{founder_bg}")
                financing = company_info.get("financing_info", "")
                if financing:
                    extras.append(f"- 💰 融资信息：{financing}")
                findings = company_info.get("key_findings", "")
                if findings:
                    extras.append(f"- 💡 补充发现：{findings}")
                if extras:
                    user_content += "\n\n### Step 0 补充信息（网络搜索获取）\n" + "\n".join(extras)
                return user_content
            except (KeyError, IndexError) as e:
                logger.warning(f"User prompt 模板填充失败: {e}，回退到行内格式")

        # 回退：行内构建
        parts = [
            f"**目标企业数据：**",
            f"- 企业名称：{name}",
            f"- 行业：{industry}",
            f"- 地区：{region}",
            f"- 注册地址：{address}",
        ]
        # Step 0 补充的实际地址
        actual_addr = company_info.get("actual_address", "")
        if actual_addr:
            parts.append(f"- ⚠️ 实际办公地址：{actual_addr}（与注册地址不同，需同时搜两个区的政策）")

        parts.extend([
            f"- 经营范围：{business_scope}",
            f"- 注册资本：{registered_capital}",
            f"- 成立时间：{founded}",
            f"- 员工规模：{employees}",
            f"- 企业标签：{tags}",
            f"- 知识产权：发明专利 {ip_invention}件，实用新型 {ip_utility}件，软著 {ip_software}件",
            f"- 股东信息：\n{shareholders_text}",
            f"- 参保人数历史：\n{headcount_text}",
            f"- 风险信息：{risk_info}",
        ])

        # Step 0 补充的其他字段
        core_products = company_info.get("core_products", "")
        if core_products:
            parts.append(f"- 🔬 核心产品/技术路线：{core_products}")
        certifications = company_info.get("certifications", [])
        if certifications:
            parts.append(f"- 🏅 已获资质：{', '.join(certifications)}")
        founder_bg = company_info.get("founder_background", "")
        if founder_bg:
            parts.append(f"- 👤 创始人背景：{founder_bg}")
        financing = company_info.get("financing_info", "")
        if financing:
            parts.append(f"- 💰 融资信息：{financing}")
        findings = company_info.get("key_findings", "")
        if findings:
            parts.append(f"- 💡 补充发现：{findings}")

        parts.append(f"\n请根据专家认知框架，对该企业进行特征逆向工程并生成搜索策略。")
        return "\n".join(parts)

    # ─────────────────────────────────────
    # Step 0: 企业信息补全（搜索企业本身）
    # ─────────────────────────────────────

    def _enrich_company_info(self, company_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        在搜政策之前，先用 2 次 web search 搜企业本身，补全企查查给不了的信息：
        - 实际办公地址（可能≠注册地址）
        - 核心产品/技术路线
        - 已获资质（高企/专精特新/科技型中小企业）
        - 创始人背景（海归/博士/院士）
        - 融资轮次
        """
        name = company_info.get("name", "")
        if not name:
            return company_info

        self._log(f"\n{'─'*30}")
        self._log(f"🔎 Step 0: 企业信息补全 — {name}")
        self._log(f"{'─'*30}")

        from web_search_worker import WebSearchWorker
        worker = WebSearchWorker()

        search_queries = [
            f'"{name}" 官网 产品 融资 技术',
            f'"{name}" 高新技术 专精特新 获奖 补贴 认定',
        ]

        raw_texts = []
        for i, q in enumerate(search_queries, 1):
            if self._is_timeout():
                break
            self._log(f"   🔍 [{i}/{len(search_queries)}] 搜索企业信息: {q}")
            try:
                result = worker.search(q)
                for p in result.policies:
                    raw_texts.append(f"标题: {p.title}\n摘要: {p.summary}\n来源: {p.source}")
                if result.raw_answer:
                    raw_texts.append(result.raw_answer[:2000])
                self._log(f"   ✅ 获取到 {len(result.policies)} 条信息, 耗时 {result.duration}s")
            except Exception as e:
                self._log(f"   ⚠️ 搜索失败: {e}")

            if i < len(search_queries) and self.request_delay > 0:
                time.sleep(self.request_delay)

        worker.close()

        if not raw_texts:
            self._log(f"   ⚠️ 未获取到企业补充信息，跳过补全")
            return company_info

        # AI 解析补充信息
        self._log(f"   🧠 AI 解析企业补充信息...")
        enrich_prompt = (
            "你是一个企业信息分析专家。根据搜索到的关于该企业的信息，提取以下补充数据。\n"
            "只提取在搜索结果中明确提到的信息，不要推测。没有信息的字段填null。\n\n"
            "输出严格 JSON：\n"
            "{\n"
            '  "actual_address": "实际办公/生产地址（如与注册地址不同）",\n'
            '  "core_products": "核心产品或技术路线（如800G光模块、VCSEL芯片等）",\n'
            '  "certifications": ["已获资质列表，如高新技术企业、科技型中小企业、专精特新等"],\n'
            '  "founder_background": "创始人/核心团队背景（如海归博士、院士等）",\n'
            '  "financing_info": "融资信息（如B轮、估值等）",\n'
            '  "key_findings": "其他对政策匹配有价值的发现（1-2句话）"\n'
            "}\n"
        )

        search_text = "\n\n---\n\n".join(raw_texts[:10])
        user_content = (
            f"【企业名称】{name}\n"
            f"【注册地址】{company_info.get('address', '未知')}\n"
            f"【行业】{company_info.get('industry', '未知')}\n\n"
            f"【搜索到的信息】\n{search_text}"
        )

        try:
            enriched = self._ai_call(enrich_prompt, user_content)
            enriched_info = dict(company_info)

            actual_addr = enriched.get("actual_address")
            if actual_addr and actual_addr != "null" and actual_addr != company_info.get("address", ""):
                enriched_info["actual_address"] = actual_addr
                self._log(f"   📍 补充实际地址: {actual_addr}")

            core_products = enriched.get("core_products")
            if core_products and core_products != "null":
                enriched_info["core_products"] = core_products
                self._log(f"   🔬 核心产品: {core_products}")

            certs = enriched.get("certifications", [])
            if certs and certs != [None] and certs != ["null"]:
                certs = [c for c in certs if c and c != "null"]
                if certs:
                    enriched_info["certifications"] = certs
                    self._log(f"   🏅 已获资质: {', '.join(certs)}")

            founder = enriched.get("founder_background")
            if founder and founder != "null":
                enriched_info["founder_background"] = founder
                self._log(f"   👤 创始人: {founder}")

            financing = enriched.get("financing_info")
            if financing and financing != "null":
                enriched_info["financing_info"] = financing
                self._log(f"   💰 融资: {financing}")

            findings = enriched.get("key_findings")
            if findings and findings != "null":
                enriched_info["key_findings"] = findings
                self._log(f"   💡 发现: {findings}")

            self._log(f"   ✅ 企业信息补全完成 (耗时 {self._elapsed()}s)")
            return enriched_info

        except Exception as e:
            self._log(f"   ⚠️ AI 解析失败: {e}，使用原始信息继续")
            return company_info

    # ─────────────────────────────────────
    # Step 1: AI 拆分任务（专家特征工程）
    # ─────────────────────────────────────

    def plan(self, company_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI 分析企业信息（专家特征工程），生成搜索任务计划。
        不执行搜索，仅返回任务列表。

        支持扩展字段：address, business_scope, shareholders, ip,
                     headcount_history, risk_info 等。
        缺失字段会自动跳过对应维度分析。

        Returns:
            {"feature_engineering": {...}, "analysis": "...", "tasks": [...], "compliance_veto": {...}}
        """
        self._log(f"🧠 AI 正在分析企业信息（专家特征工程）: {company_info.get('name', '?')}")

        system = _build_plan_system_prompt()
        user_content = self._build_user_content(company_info)

        plan = self._ai_call(system, user_content)

        # ── 合规熔断检查 ──
        veto = plan.get("compliance_veto", {})
        if veto and not veto.get("passed", True):
            risk_level = veto.get("risk_level", "unknown")
            detail = veto.get("detail", "")
            self._log(f"🚫 合规熔断触发！风险等级: {risk_level}")
            self._log(f"   原因: {detail}")
            if risk_level == "blocked":
                self._log(f"   ⛔ 企业处于'政策绝缘'状态，仅搜索信用修复路径")

        # ── 特征工程日志 ──
        fe = plan.get("feature_engineering", {})
        if fe:
            self._log(f"\n📐 特征逆向工程结果:")
            for dim, result in fe.items():
                dim_icon = {"spatial": "📍", "industry_chain": "🏭", "identity": "🏷️",
                           "hr_dynamics": "👥", "compliance": "⚖️",
                           "tax_financial": "📊", "talent_incentive": "🏆"}.get(dim, "📌")
                self._log(f"   {dim_icon} {dim}: {result}")

        # ── 差距分析日志 ──
        gap = plan.get("gap_analysis", {})
        if gap:
            self._log(f"\n📊 维度差距分析:")
            for dim, result in gap.items():
                self._log(f"   → {dim}: {result}")

        # ── 搜索任务日志 ──
        analysis = plan.get("analysis", "")
        tasks = plan.get("tasks", [])
        self._log(f"\n📋 AI 分析: {analysis}")
        self._log(f"📋 生成 {len(tasks)} 个搜索任务:")
        for i, t in enumerate(tasks, 1):
            priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(t.get("priority", ""), "⚪")
            dim = t.get("dimension", "?")
            layer = t.get("layer", "?")
            self._log(f"   {i}. [{dim}→{layer}] {priority_icon} {t.get('search_term', '?')}")
            self._log(f"      原因: {t.get('reason', '')}")
            if t.get("focus_hints"):
                self._log(f"      🔎 关注: {t['focus_hints']}")

        return plan

    # ─────────────────────────────────────
    # Step 2: 执行 Web Search
    # ─────────────────────────────────────

    def _run_web_searches(self, tasks: List[Dict]) -> List[WorkerResult]:
        """
        逐个执行 web search 任务，收集结果。
        支持请求间隔（避免 429）和超时检查。
        """
        from web_search_worker import WebSearchWorker

        worker = WebSearchWorker()
        results = []

        for i, task in enumerate(tasks, 1):
            # 超时检查
            if self._is_timeout():
                self._log(f"   ⏰ 时间预算用尽（已 {self._elapsed()}s），跳过剩余 {len(tasks)-i+1} 个任务")
                break

            term = task.get("search_term", "")
            layer = task.get("layer", "?")
            remaining = round(self._time_remaining())
            self._log(f"🔍 [{i}/{len(tasks)}] Web搜索 [{layer}]: {term}  (剩余 {remaining}s)")

            try:
                result = worker.search(term)
                result.worker = f"web_search({layer})"
                results.append(result)
                self._log(f"   ✅ 找到 {result.policy_count} 条政策, 耗时 {result.duration}s")
            except Exception as e:
                self._log(f"   ❌ 搜索失败: {e}")
                results.append(WorkerResult(query=term, worker=f"web_search({layer})", error=str(e)))

            # 请求间隔（避免 429），最后一个不等
            if i < len(tasks) and self.request_delay > 0:
                time.sleep(self.request_delay)

        worker.close()
        return results

    # ─────────────────────────────────────
    # Step 3: AI 评估 → 是否需要 browse use
    # ─────────────────────────────────────

    def _evaluate_results(self, all_policies: List[PolicyItem]) -> Dict[str, Any]:
        """
        AI 评估搜索结果，决定哪些需要 browse use。
        """
        if not all_policies:
            return {"evaluation": "无搜索结果", "browse_targets": [], "skip_reasons": []}

        self._log(f"🧠 AI 正在评估 {len(all_policies)} 条搜索结果...")

        # 构建评估输入
        items_text = []
        for i, p in enumerate(all_policies, 1):
            items_text.append(
                f"{i}. 标题: {p.title}\n"
                f"   URL: {p.url}\n"
                f"   摘要: {p.summary or '无'}\n"
                f"   扶持: {p.support or '无'}\n"
                f"   来源: {p.source or '无'}"
            )

        user_content = f"以下是 web search 返回的政策条目，请评估哪些需要用浏览器深度抓取：\n\n" + "\n\n".join(items_text)

        evaluation = self._ai_call(EVALUATE_SYSTEM_PROMPT, user_content)

        targets = evaluation.get("browse_targets", [])
        self._log(f"📋 评估完成: {evaluation.get('evaluation', '')}")
        self._log(f"   需要深度抓取: {len(targets)} 条")
        for t in targets:
            self._log(f"   → {t.get('title', '?')} ({t.get('reason', '')})")

        return evaluation

    # ─────────────────────────────────────
    # Step 6: AI 打分排序 + 有效期
    # ─────────────────────────────────────

    def _score_policies(self, company_info: Dict[str, Any], policies: List[PolicyItem]) -> List[PolicyItem]:
        """
        AI 为每条政策打分（相关度）并补充有效期，按分数排序。
        """
        if not policies:
            return policies

        self._log(f"\n{'─'*30}")
        self._log(f"📊 AI 打分排序（{len(policies)} 条政策）")
        self._log(f"{'─'*30}")

        # 构建输入
        items_text = []
        for i, p in enumerate(policies, 1):
            items_text.append(
                f"{i}. [{p.layer or '?'}] {p.title}\n"
                f"   摘要: {(p.summary or '')[:100]}\n"
                f"   扶持: {p.support or '无'}\n"
                f"   发布日期: {p.date or '未知'}\n"
                f"   有效期: {p.validity or '未知'}\n"
                f"   申报截止: {p.application_deadline or '未知'}"
            )

        user_content = (
            f"【企业信息】\n"
            f"名称: {company_info.get('name', '?')}\n"
            f"行业: {company_info.get('industry', '?')}\n"
            f"地区: {company_info.get('region', '?')}\n"
            f"标签: {', '.join(company_info.get('tags', []))}\n\n"
            f"【待评分政策（{len(policies)} 条）】\n" + "\n\n".join(items_text)
        )

        try:
            result = self._ai_call(_build_scoring_system_prompt(), user_content)
            scored = result.get("scored_policies", [])

            for item in scored:
                idx = item.get("index", 0) - 1
                if 0 <= idx < len(policies):
                    policies[idx].relevance = item.get("relevance", 0)
                    policies[idx].score_amount = item.get("score_amount", 0)
                    policies[idx].score_exclusivity = item.get("score_exclusivity", 0)
                    policies[idx].score_feasibility = item.get("score_feasibility", 0)
                    policies[idx].score_urgency = item.get("score_urgency", 0)
                    policies[idx].score_sustainability = item.get("score_sustainability", 0)
                    policies[idx].score_reason = item.get("reason", "")
                    policies[idx].validity = item.get("validity", "")
                    policies[idx].amount = item.get("amount", "")
                    policies[idx].amount_level = item.get("amount_level", "")

            # 按综合分排序（高→低）
            policies.sort(key=lambda p: p.relevance, reverse=True)

            # 日志 — 显示5维度评分
            for p in policies:
                score_bar = "█" * (p.relevance // 10) + "░" * (10 - p.relevance // 10)
                lvl = p.amount_level or '?'
                self._log(
                    f"   {p.relevance:3d}分 {score_bar} [{p.layer or '?'}] {p.title[:30]}  "
                    f"💰{lvl}:{p.amount or '?'}  📅{p.validity or '?'}  "
                    f"[💰{p.score_amount} 🎯{p.score_exclusivity} ✅{p.score_feasibility} ⏰{p.score_urgency} 🔄{p.score_sustainability}]"
                )

        except Exception as e:
            self._log(f"   ⚠️ 打分失败（不影响结果）: {e}")

        return policies

    # ─────────────────────────────────────
    # Step 3b: 回路评估 — 判断是否需要补充搜索
    # ─────────────────────────────────────

    def _review_round(
        self,
        company_info: Dict[str, Any],
        round_num: int,
        all_policies: List[PolicyItem],
        search_history: List[str],
        feature_engineering: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        AI 评审当前轮次的搜索结果，判断是否需要补充搜索。
        使用五维度+四层双重评估。

        Args:
            company_info:        企业信息
            round_num:           当前轮次
            all_policies:        已收集到的所有政策
            search_history:      已使用过的搜索词
            feature_engineering: plan() 阶段的特征工程结果（可选）

        Returns:
            {"overall_quality": "good|fair|poor", "need_more_search": bool, "retry_tasks": [...]}
        """
        self._log(f"\n🔄 第 {round_num} 轮评审（已有 {len(all_policies)} 条政策，已用 {self._elapsed()}s）")

        # 按 layer 统计结果数
        layer_counts: Dict[str, int] = {}
        for p in all_policies:
            worker = getattr(p, 'layer', '') or ''
            if worker:
                layer_counts[worker] = layer_counts.get(worker, 0) + 1

        system = _build_round_review_system_prompt()

        # 构建已有结果摘要（含日期和有效期）
        results_summary = []
        for i, p in enumerate(all_policies, 1):
            date_info = p.date or '日期未知'
            validity_info = f" | 有效期:{p.validity}" if p.validity else ""
            deadline_info = f" | 申报截止:{p.application_deadline}" if p.application_deadline else ""
            support_info = p.support or (p.summary[:50] if p.summary else '无摘要')
            results_summary.append(f"{i}. [{getattr(p, 'layer', '?')}] {p.title} — {date_info}{validity_info}{deadline_info} — {support_info}")

        # 使用丰富的企业信息（与 plan 阶段一致）
        enterprise_summary = self._build_user_content(company_info)

        user_content = (
            f"【企业特征分析】\n{enterprise_summary}\n\n"
        )

        # 附加 plan 阶段的特征工程结果
        if feature_engineering:
            user_content += (
                f"【特征工程结果（来自 plan 阶段）】\n"
                f"{json.dumps(feature_engineering, ensure_ascii=False, indent=2)}\n\n"
            )

        user_content += (
            f"【当前日期】{datetime.now().strftime('%Y-%m-%d')}\n"
            f"【当前轮次】第 {round_num} 轮\n"
            f"【时间剩余】{round(self._time_remaining())}s\n"
            f"【已用搜索词】\n" + "\n".join(f"  - {s}" for s in search_history) + "\n\n"
            f"【已搜到的政策（{len(all_policies)} 条）】\n" + "\n".join(results_summary) + "\n\n"
            f"请评审搜索质量，判断是否需要补充搜索。如果时间不足30秒，请设置 need_more_search=false。"
        )

        review = self._ai_call(system, user_content)

        quality = review.get("overall_quality", "?")
        quality_icon = {"good": "🟢", "fair": "🟡", "poor": "🔴"}.get(quality, "⚪")
        self._log(f"   {quality_icon} 质量: {quality} — {review.get('quality_reason', '')}")

        # 打印维度覆盖情况
        dim_cov = review.get("dimension_coverage", {})
        if dim_cov:
            self._log(f"   ── 维度覆盖 ──")
            for dim, info in dim_cov.items():
                status = info.get("status", "?")
                s_icon = {"sufficient": "✅", "insufficient": "⚠️", "missing": "❌", "not_applicable": "➖"}.get(status, "?")
                self._log(f"   {s_icon} {dim}: {status} ({info.get('count', '?')}条) {info.get('note', '')}")

        # 打印各层覆盖情况
        layer_cov = review.get("layer_coverage", {})
        for layer, info in layer_cov.items():
            status = info.get("status", "?")
            s_icon = {"sufficient": "✅", "insufficient": "⚠️", "missing": "❌"}.get(status, "?")
            self._log(f"   {s_icon} {layer}: {status} ({info.get('count', '?')}条) {info.get('note', '')}")

        # 打印时效性评估
        timeliness = review.get("timeliness", {})
        if timeliness:
            t_status = timeliness.get("status", "?")
            t_icon = {"good": "✅", "poor": "⚠️"}.get(t_status, "?")
            self._log(f"   {t_icon} 时效性: {t_status} (当年{timeliness.get('current_year_count', '?')}条, 过期{timeliness.get('outdated_count', '?')}条) {timeliness.get('note', '')}")

        need_more = review.get("need_more_search", False)
        retry_tasks = review.get("retry_tasks", [])
        if need_more and retry_tasks:
            self._log(f"   🔄 需要补充搜索 {len(retry_tasks)} 个任务:")
            for t in retry_tasks:
                dim = t.get("dimension", "?")
                layer = t.get("layer", "?")
                self._log(f"      → [{dim}→{layer}] {t.get('search_term', '?')} ({t.get('reason', '')})")
        else:
            self._log(f"   ✅ 搜索质量{'' if quality == 'good' else '基本'}满足要求，不再补充")

        return review

    # ─────────────────────────────────────
    # Step 4: Browse Use 深度抓取
    # ─────────────────────────────────────

    async def _run_browse_use(self, targets: List[Dict]) -> List[PolicyItem]:
        """
        对需要深度抓取的目标执行 browse use。
        """
        if not targets:
            return []

        from browser_use_worker import BrowserUseWorker
        worker = BrowserUseWorker()
        results = []

        for i, target in enumerate(targets, 1):
            url = target.get("url", "")
            title = target.get("title", "?")
            self._log(f"🌐 [{i}/{len(targets)}] Browse Use 深度抓取: {title}")

            try:
                task = (
                    f"请访问以下URL并提取完整的政策信息：\n"
                    f"URL: {url}\n"
                    f"标题: {title}\n\n"
                    f"提取：政策全文摘要、扶持金额/比例、申报条件、截止日期、PDF下载链接。"
                )
                result = worker.search(task)
                results.extend(result.policies)
                self._log(f"   ✅ 提取到 {result.policy_count} 条详细政策")
            except Exception as e:
                self._log(f"   ❌ 深度抓取失败: {e}")

        return results

    # ─────────────────────────────────────
    # Step 5: 去重
    # ─────────────────────────────────────

    @staticmethod
    def deduplicate(policies: List[PolicyItem]) -> List[PolicyItem]:
        """
        去重逻辑：按 (标题, URL) 去重，保留信息更完整的版本。
        """
        seen: Dict[str, PolicyItem] = {}  # key → PolicyItem

        for p in policies:
            title = p.title.strip()
            url = p.url.strip().rstrip("/")
            key = f"{title}||{url}"

            if key in seen:
                # 保留摘要更长的版本
                existing = seen[key]
                if len(p.summary or "") > len(existing.summary or ""):
                    seen[key] = p
                # 补充缺失字段
                if p.pdf_url and not existing.pdf_url:
                    seen[key].pdf_url = p.pdf_url
                if p.support and not existing.support:
                    seen[key].support = p.support
                if p.full_text and not existing.full_text:
                    seen[key].full_text = p.full_text
            else:
                seen[key] = p

        return list(seen.values())

    # ─────────────────────────────────────
    # 主流程
    # ─────────────────────────────────────

    async def run(self, company_info: Dict[str, Any], skip_browse_use: bool = False) -> WorkerResult:
        """
        完整执行 orchestrator 流程（带评估反馈回路）。

        流程：
            Round 1: AI拆分 → Web Search → AI评审
            Round 2: (如需) AI 给出补充搜索词 → Web Search → AI评审
            Round N: ... 直到质量 good 或超时/轮次用尽
            最后:    AI 评估 browse use → 执行(可选) → 合并去重

        Args:
            company_info:    企查查企业信息
            skip_browse_use: 跳过 browse use（用于快速测试）

        Returns:
            WorkerResult（合并后的最终结果）
        """
        self._start_time = time.time()
        company_name = company_info.get("name", "未知企业")

        self._log(f"{'='*50}")
        self._log(f"🚀 Orchestrator 启动: {company_name}")
        self._log(f"   时间预算: {self.time_budget}s | 最大轮次: {self.max_rounds} | 请求间隔: {self.request_delay}s")
        self._log(f"{'='*50}")

        # ── Step 0: 企业信息补全 ──
        enriched_info = self._enrich_company_info(company_info)

        # ── Step 1: AI 拆分任务（专家特征工程） ──
        plan = self.plan(enriched_info)
        tasks = plan.get("tasks", [])

        # 保存特征工程结果，供回路评估使用
        feature_engineering = plan.get("feature_engineering", {})

        # ── 合规熔断检查 ──
        veto = plan.get("compliance_veto", {})
        if veto and not veto.get("passed", True) and veto.get("risk_level") == "blocked":
            self._log("⛔ 企业处于政策绝缘状态，仅保留信用修复相关搜索任务")
            # 只保留合规维度的任务
            tasks = [t for t in tasks if t.get("dimension") == "合规"]
            if not tasks:
                return WorkerResult(
                    query=company_name,
                    error="企业存在严重失信记录，暂无法匹配政策。建议先进行信用修复。",
                    worker="orchestrator",
                    duration=round(time.time() - self._start_time, 1),
                )

        if not tasks:
            self._log("⚠️ AI 未生成任何搜索任务")
            return WorkerResult(query=company_name, error="AI 未生成搜索任务")

        # ── 搜索回路 ──
        all_policies: List[PolicyItem] = []
        all_sources: List[str] = []
        total_tokens: Dict[str, int] = {}
        search_history: List[str] = []
        round_num = 0

        current_tasks = tasks
        while round_num < self.max_rounds:
            round_num += 1

            # 超时检查
            if self._is_timeout():
                self._log(f"\n⏰ 时间预算用尽（{self._elapsed()}s），停止搜索")
                break

            self._log(f"\n{'─'*30}")
            self._log(f"📡 第 {round_num} 轮 Web Search（{len(current_tasks)} 个任务，已用 {self._elapsed()}s）")
            self._log(f"{'─'*30}")

            # 执行 Web Search
            loop = asyncio.get_event_loop()
            web_results = await loop.run_in_executor(None, self._run_web_searches, current_tasks)

            # 汇总本轮结果
            round_policies = []
            for r in web_results:
                # 给每条 policy 打上 layer 标记
                layer_tag = r.worker.replace("web_search(", "").rstrip(")")
                for p in r.policies:
                    p.layer = layer_tag
                round_policies.extend(r.policies)
                all_sources.extend(r.sources)
                if r.token_usage:
                    for k, v in r.token_usage.items():
                        total_tokens[k] = total_tokens.get(k, 0) + v

            # 记录搜索词
            for t in current_tasks:
                search_history.append(t.get("search_term", ""))

            all_policies.extend(round_policies)
            self._log(f"\n📊 第 {round_num} 轮: +{len(round_policies)} 条, 累计 {len(all_policies)} 条 (已用 {self._elapsed()}s)")

            # 最后一轮不评审
            if round_num >= self.max_rounds:
                self._log(f"\n🛑 已达最大轮次 ({self.max_rounds})，结束搜索")
                break

            # 超时检查（评审也需要时间）
            if self._time_remaining() < 45:
                self._log(f"\n⏰ 剩余时间不足45s，跳过评审")
                break

            # ── AI 评审回路 ──
            review = self._review_round(company_info, round_num, all_policies, search_history, feature_engineering)

            if not review.get("need_more_search", False):
                self._log(f"\n✅ 搜索质量达标，结束搜索回路")
                break

            # 准备下一轮任务
            retry_tasks = review.get("retry_tasks", [])
            if not retry_tasks:
                self._log(f"\n✅ 无补充任务，结束搜索回路")
                break

            # 去掉已搜过的词
            new_tasks = [t for t in retry_tasks if t.get("search_term", "") not in search_history]
            if not new_tasks:
                self._log(f"\n✅ 补充搜索词都已搜过，结束搜索回路")
                break

            current_tasks = new_tasks

        # ── Step 3: AI 评估是否需要 browse use ──
        browse_policies = []
        if not skip_browse_use and all_policies:
            if self._time_remaining() > 90:  # browse use 至少需要 90s
                self._log(f"\n{'─'*30}")
                self._log(f"🧠 AI 评估搜索质量（是否需要 Browse Use）")
                self._log(f"{'─'*30}")

                evaluation = self._evaluate_results(all_policies)
                targets = evaluation.get("browse_targets", [])

                if targets and self._time_remaining() > 90:
                    self._log(f"\n{'─'*30}")
                    self._log(f"🌐 Browse Use 深度抓取（{len(targets)} 个目标）")
                    self._log(f"{'─'*30}")
                    browse_policies = await self._run_browse_use(targets)
                elif targets:
                    self._log(f"\n⏰ 剩余时间不足90s，跳过 Browse Use（需深度抓取 {len(targets)} 条）")
            else:
                self._log(f"\n⏰ 剩余时间不足90s，跳过 Browse Use 评估")
        elif skip_browse_use:
            self._log("\n⏭️ 跳过 Browse Use（skip_browse_use=True）")

        # ── Step 5: 合并 + 去重 ──
        self._log(f"\n{'─'*30}")
        self._log(f"🔗 合并与去重")
        self._log(f"{'─'*30}")

        combined = all_policies + browse_policies
        final = self.deduplicate(combined)
        self._log(f"   合并前: {len(combined)} 条 → 去重后: {len(final)} 条")

        # ── Step 6: AI 打分排序 + 有效期 ──
        if final and not self._is_timeout():
            final = self._score_policies(company_info, final)

        elapsed = round(time.time() - self._start_time, 1)

        # 构建最终结果
        result = WorkerResult(
            query=f"{company_name} 政策搜索",
            policies=final,
            sources=list(set(all_sources)),
            worker="orchestrator",
            duration=elapsed,
            token_usage=total_tokens,
        )

        self._log(f"\n{'='*50}")
        self._log(f"✅ Orchestrator 完成!")
        self._log(f"   企业: {company_name}")
        self._log(f"   政策: {result.policy_count} 条")
        self._log(f"   轮次: {round_num}")
        self._log(f"   耗时: {elapsed}s")
        self._log(f"{'='*50}")

        return result
