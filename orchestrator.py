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
from typing import List, Dict, Any, Optional, Callable

from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.web_search")

from models import PolicyItem, WorkerResult
from policy_categories import get_layers_reference

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# AI 思考 Prompt
# ─────────────────────────────────────────────

PLAN_SYSTEM_PROMPT = """你是一个政策搜索调度专家。你的任务是根据企业信息，拆分出最有效的搜索任务列表。

{layers_ref}

【规则】
1. 根据企业的行业、地区、发展阶段、标签等信息，判断哪些政策层与该企业最相关
2. 为每个相关层生成 1-2 个精准的搜索关键词（尽量具体，包含地区和行业）
3. 如果企业有特殊标签（如"高新技术"、"专精特新"），增加针对性搜索
4. 搜索词应该适合在 Bing/百度 上直接搜索，不要太长
5. 输出严格 JSON，不要输出其他文字

【输出格式】
{{
  "analysis": "对企业的简要分析（1-2句话）",
  "tasks": [
    {{
      "layer": "基础层|发展层|人才层|荣誉层",
      "search_term": "搜索关键词",
      "priority": "high|medium|low",
      "reason": "为什么搜这个（简短）"
    }}
  ]
}}
"""

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

SCORING_SYSTEM_PROMPT = """你是一个政策匹配评分专家。根据企业信息，为每条政策打分并补充有效期和金额范围。

【评分规则（0-100分）】
- 90-100: 政策完全匹配企业行业+地区+发展阶段，扶持力度大
- 70-89:  政策高度相关，但可能不是完全对口
- 50-69:  有一定参考价值，但匹配度一般
- 30-49:  关联较弱，仅供了解
- 0-29:   基本不相关

【有效期判断】
- 如果摘要/标题中有明确年份范围（如"2024-2025"），提取为有效期
- 如果是长期政策（如税收优惠），标注"长期有效"
- 如果无法判断，标注"请查原文确认"

【金额提取】
- 从摘要/扶持内容中提取关键金额，用简短格式表达
- 例如："最高20万"、"最高1000万"、"10%-20%补贴"、"税率减半"、"50万奖励"
- 如果有多档金额，取最高档
- 如果无明确金额，填"见原文"

输出严格 JSON：
{{
  "scored_policies": [
    {{
      "index": 1,
      "relevance": 85,
      "validity": "2025-12-31",
      "amount": "最高20万",
      "reason": "评分理由（简短）"
    }}
  ]
}}
"""

# ── 回路评估 Prompt ──

ROUND_REVIEW_SYSTEM_PROMPT = """你是一个政策搜索质量评审专家。你刚完成了一轮搜索，现在需要判断结果质量。

{layers_ref}

【你的任务】
根据企业信息和已搜到的政策结果，回答以下问题：
1. 哪些业务层（基础层/发展层/人才层/荣誉层）的结果已经足够？
2. 哪些业务层的结果不足或缺失，需要补充搜索？
3. 之前的搜索词是否有效？如果无效，给出改进后的搜索词
4. 是否有明显遗漏的政策方向？

【判断标准】
- 每个相关层至少有 1-2 条有效政策（含 URL 和扶持内容）算"足够"
- 某层 0 条结果，或全部结果摘要空白 → "不足"
- 搜索词太泛（结果不相关）或太窄（0 结果）→ 需要调整

输出严格 JSON：
{{
  "overall_quality": "good|fair|poor",
  "quality_reason": "整体质量判断原因（1-2句话）",
  "layer_coverage": {{
    "基础层": {{"status": "sufficient|insufficient|missing", "count": 0, "note": "..."}},
    "发展层": {{"status": "sufficient|insufficient|missing", "count": 0, "note": "..."}},
    "人才层": {{"status": "sufficient|insufficient|missing", "count": 0, "note": "..."}},
    "荣誉层": {{"status": "sufficient|insufficient|missing", "count": 0, "note": "..."}}
  }},
  "need_more_search": true,
  "retry_tasks": [
    {{
      "layer": "需要补充的层",
      "search_term": "改进后的搜索词",
      "reason": "为什么需要重新搜索"
    }}
  ]
}}
"""


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
        time_budget: float = 180.0,
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

    def plan(self, company_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI 分析企业信息，生成搜索任务计划。
        不执行搜索，仅返回任务列表。

        Args:
            company_info: 企查查信息
                {
                    "name": "企业名称",
                    "industry": "行业",
                    "region": "地区",
                    "tags": ["高新技术", ...],
                    "registered_capital": "1000万",  # 可选
                    "employees": "50-100",            # 可选
                    "founded": "2020",                # 可选
                }

        Returns:
            {"analysis": "...", "tasks": [...]}
        """
        self._log(f"🧠 AI 正在分析企业信息: {company_info.get('name', '?')}")

        layers_ref = get_layers_reference()
        system = PLAN_SYSTEM_PROMPT.format(layers_ref=layers_ref)

        user_content = (
            f"企业信息：\n"
            f"- 名称：{company_info.get('name', '未知')}\n"
            f"- 行业：{company_info.get('industry', '未知')}\n"
            f"- 地区：{company_info.get('region', '未知')}\n"
            f"- 标签：{', '.join(company_info.get('tags', []))}\n"
        )
        # 附加可选信息
        if company_info.get("registered_capital"):
            user_content += f"- 注册资本：{company_info['registered_capital']}\n"
        if company_info.get("employees"):
            user_content += f"- 员工规模：{company_info['employees']}\n"
        if company_info.get("founded"):
            user_content += f"- 成立时间：{company_info['founded']}\n"

        user_content += "\n请为这家企业生成搜索任务计划。"

        plan = self._ai_call(system, user_content)

        # 日志
        analysis = plan.get("analysis", "")
        tasks = plan.get("tasks", [])
        self._log(f"📋 AI 分析: {analysis}")
        self._log(f"📋 生成 {len(tasks)} 个搜索任务:")
        for i, t in enumerate(tasks, 1):
            priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(t.get("priority", ""), "⚪")
            self._log(f"   {i}. [{t.get('layer', '?')}] {priority_icon} {t.get('search_term', '?')}")
            self._log(f"      原因: {t.get('reason', '')}")

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
                f"   日期: {p.date or '未知'}"
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
            result = self._ai_call(SCORING_SYSTEM_PROMPT, user_content)
            scored = result.get("scored_policies", [])

            for item in scored:
                idx = item.get("index", 0) - 1
                if 0 <= idx < len(policies):
                    policies[idx].relevance = item.get("relevance", 0)
                    policies[idx].validity = item.get("validity", "")
                    policies[idx].amount = item.get("amount", "")

            # 按分数排序（高→低）
            policies.sort(key=lambda p: p.relevance, reverse=True)

            # 日志
            for p in policies:
                score_bar = "█" * (p.relevance // 10) + "░" * (10 - p.relevance // 10)
                self._log(f"   {p.relevance:3d}分 {score_bar} [{p.layer or '?'}] {p.title[:30]}  💰{p.amount or '?'}  📅{p.validity or '?'}")

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
    ) -> Dict[str, Any]:
        """
        AI 评审当前轮次的搜索结果，判断是否需要补充搜索。

        Args:
            company_info:   企业信息
            round_num:      当前轮次
            all_policies:   已收集到的所有政策
            search_history: 已使用过的搜索词

        Returns:
            {"overall_quality": "good|fair|poor", "need_more_search": bool, "retry_tasks": [...]}
        """
        self._log(f"\n🔄 第 {round_num} 轮评审（已有 {len(all_policies)} 条政策，已用 {self._elapsed()}s）")

        # 按 layer 统计结果数
        layer_counts: Dict[str, int] = {}
        for p in all_policies:
            # 从 worker 字段中提取 layer
            worker = getattr(p, 'layer', '') or ''
            if worker:
                layer_counts[worker] = layer_counts.get(worker, 0) + 1

        layers_ref = get_layers_reference()
        system = ROUND_REVIEW_SYSTEM_PROMPT.format(layers_ref=layers_ref)

        # 构建已有结果摘要
        results_summary = []
        for i, p in enumerate(all_policies, 1):
            results_summary.append(f"{i}. [{getattr(p, 'layer', '?')}] {p.title} — {p.support or p.summary[:50] if p.summary else '无摘要'}")

        user_content = (
            f"【企业信息】\n"
            f"名称: {company_info.get('name', '?')}\n"
            f"行业: {company_info.get('industry', '?')}\n"
            f"地区: {company_info.get('region', '?')}\n"
            f"标签: {', '.join(company_info.get('tags', []))}\n\n"
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

        # 打印各层覆盖情况
        layer_cov = review.get("layer_coverage", {})
        for layer, info in layer_cov.items():
            status = info.get("status", "?")
            s_icon = {"sufficient": "✅", "insufficient": "⚠️", "missing": "❌"}.get(status, "?")
            self._log(f"   {s_icon} {layer}: {status} ({info.get('count', '?')}条) {info.get('note', '')}")

        need_more = review.get("need_more_search", False)
        retry_tasks = review.get("retry_tasks", [])
        if need_more and retry_tasks:
            self._log(f"   🔄 需要补充搜索 {len(retry_tasks)} 个任务:")
            for t in retry_tasks:
                self._log(f"      → [{t.get('layer', '?')}] {t.get('search_term', '?')} ({t.get('reason', '')})")
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

        # ── Step 1: AI 拆分任务 ──
        plan = self.plan(company_info)
        tasks = plan.get("tasks", [])

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
            if self._time_remaining() < 30:
                self._log(f"\n⏰ 剩余时间不足30s，跳过评审")
                break

            # ── AI 评审回路 ──
            review = self._review_round(company_info, round_num, all_policies, search_history)

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
            if self._time_remaining() > 60:  # browse use 至少需要 60s
                self._log(f"\n{'─'*30}")
                self._log(f"🧠 AI 评估搜索质量（是否需要 Browse Use）")
                self._log(f"{'─'*30}")

                evaluation = self._evaluate_results(all_policies)
                targets = evaluation.get("browse_targets", [])

                if targets and self._time_remaining() > 60:
                    self._log(f"\n{'─'*30}")
                    self._log(f"🌐 Browse Use 深度抓取（{len(targets)} 个目标）")
                    self._log(f"{'─'*30}")
                    browse_policies = await self._run_browse_use(targets)
                elif targets:
                    self._log(f"\n⏰ 剩余时间不足，跳过 Browse Use（需深度抓取 {len(targets)} 条）")
            else:
                self._log(f"\n⏰ 剩余时间不足60s，跳过 Browse Use 评估")
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
