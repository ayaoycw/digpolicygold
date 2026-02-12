"""
Orchestrator 测试
=================
测试 AI 智能调度的三个关键环节：
    1. plan    — AI 拆分搜索任务（只调 AI，不执行搜索）
    2. search  — 拆分 + 执行 web search（不调 browse use）
    3. full    — 完整流程（含 browse use 评估和执行）

用法：
    # 只测试 AI 拆分（最快，验证 AI 思考逻辑）
    python test_orchestrator.py plan

    # 测试拆分 + web search（验证搜索质量）
    python test_orchestrator.py search

    # 完整流程
    python test_orchestrator.py full
"""

import asyncio
import argparse
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from orchestrator import Orchestrator
from models import PolicyItem


# ─────────────────────────────────────────────
# 测试用企业信息（模拟企查查数据）
# ─────────────────────────────────────────────

SAMPLE_COMPANIES = {
    "光通信": {
        "name": "上海智光通信科技有限公司",
        "industry": "光通信",
        "region": "上海市浦东新区",
        "tags": ["高新技术企业", "专精特新"],
        "registered_capital": "5000万",
        "employees": "100-300",
        "founded": "2018",
    },
    "AI": {
        "name": "深圳智脑人工智能有限公司",
        "industry": "人工智能",
        "region": "深圳市南山区",
        "tags": ["国家级高新技术企业", "创新企业", "独角兽"],
        "registered_capital": "1亿",
        "employees": "300-500",
        "founded": "2020",
    },
    "生物医药": {
        "name": "苏州康瑞生物医药有限公司",
        "industry": "生物医药",
        "region": "苏州市工业园区",
        "tags": ["高新技术", "临床试验"],
        "registered_capital": "2000万",
        "employees": "50-100",
        "founded": "2021",
    },
}


def print_separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# 测试 1: 仅 AI 拆分
# ─────────────────────────────────────────────

def test_plan(company_key: str = "光通信"):
    """测试 AI 拆分任务"""
    print_separator(f"测试: AI 拆分任务 ({company_key})")

    company = SAMPLE_COMPANIES.get(company_key, SAMPLE_COMPANIES["光通信"])
    print(f"企业信息: {json.dumps(company, ensure_ascii=False, indent=2)}\n")

    logs = []
    def log_cb(msg):
        logs.append(msg)
        print(msg)

    orch = Orchestrator(on_log=log_cb)
    plan = orch.plan(company)

    print(f"\n--- AI 完整输出 ---")
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    # 验证
    tasks = plan.get("tasks", [])
    print(f"\n--- 验证 ---")
    print(f"✅ 任务数: {len(tasks)}")
    assert len(tasks) > 0, "AI 应该生成至少 1 个任务"

    layers_found = set(t.get("layer", "") for t in tasks)
    print(f"✅ 覆盖层: {layers_found}")

    for t in tasks:
        assert t.get("search_term"), "每个任务必须有 search_term"
        assert t.get("layer"), "每个任务必须有 layer"
        print(f"✅ 任务格式正确: [{t['layer']}] {t['search_term']}")

    print(f"\n🎉 AI 拆分测试通过!")
    return plan


# ─────────────────────────────────────────────
# 测试 2: 拆分 + Web Search
# ─────────────────────────────────────────────

async def test_search(company_key: str = "光通信", budget: float = 180, rounds: int = 3, delay: float = 2.0):
    """测试拆分 + 执行 web search（跳过 browse use），带回路评估"""
    print_separator(f"测试: AI 拆分 + Web Search + 回路 ({company_key})")

    company = SAMPLE_COMPANIES.get(company_key, SAMPLE_COMPANIES["光通信"])
    print(f"企业: {company['name']} ({company['industry']} @ {company['region']})")
    print(f"预算: {budget}s | 最大轮次: {rounds} | 请求间隔: {delay}s\n")

    logs = []
    def log_cb(msg):
        logs.append(msg)
        print(msg)

    orch = Orchestrator(on_log=log_cb, time_budget=budget, max_rounds=rounds, request_delay=delay)
    result = await orch.run(company, skip_browse_use=True)

    # 打印结果摘要
    print(f"\n--- 结果摘要 ---")
    print(f"政策数: {result.policy_count}")
    print(f"来源数: {len(result.sources)}")
    print(f"耗时:   {result.duration}s")
    print(f"Token:  {result.token_usage}")

    if result.policies:
        print(f"\n--- 政策列表 ---")
        for i, p in enumerate(result.policies, 1):
            print(f"  {i}. {p.title}")
            print(f"     URL: {p.url}")
            if p.summary:
                print(f"     摘要: {p.summary[:80]}...")
            if p.support:
                print(f"     扶持: {p.support}")
            print()

    print(f"\n🎉 Web Search 测试完成! 找到 {result.policy_count} 条政策")
    return result


# ─────────────────────────────────────────────
# 测试 3: 完整流程
# ─────────────────────────────────────────────

async def test_full(company_key: str = "光通信", budget: float = 300, rounds: int = 3, delay: float = 2.0):
    """测试完整流程（含 browse use 评估）"""
    print_separator(f"测试: 完整流程 ({company_key})")

    company = SAMPLE_COMPANIES.get(company_key, SAMPLE_COMPANIES["光通信"])
    print(f"企业: {company['name']} ({company['industry']} @ {company['region']})")
    print(f"预算: {budget}s | 最大轮次: {rounds} | 请求间隔: {delay}s\n")

    logs = []
    def log_cb(msg):
        logs.append(msg)
        print(msg)

    orch = Orchestrator(on_log=log_cb, time_budget=budget, max_rounds=rounds, request_delay=delay)
    result = await orch.run(company, skip_browse_use=False)

    print(f"\n--- 最终结果 ---")
    print(result.to_json())
    print(f"\n🎉 完整流程测试完成! 找到 {result.policy_count} 条政策, 耗时 {result.duration}s")
    return result


# ─────────────────────────────────────────────
# 去重测试（本地，不需要 API）
# ─────────────────────────────────────────────

def test_dedup():
    """测试去重逻辑"""
    print_separator("测试: 去重逻辑")

    policies = [
        PolicyItem(title="上海市人才引进政策", url="http://gov.cn/p1", summary="短摘要"),
        PolicyItem(title="上海市人才引进政策", url="http://gov.cn/p1", summary="这是一个更长更详细的摘要内容", pdf_url="http://gov.cn/p1.pdf"),
        PolicyItem(title="上海市人才引进政策", url="http://gov.cn/p1/", summary="中等摘要"),  # URL 尾部斜杠
        PolicyItem(title="深圳市创新补贴", url="http://gov.cn/p2", summary="深圳创新补贴内容"),
        PolicyItem(title="深圳市创新补贴", url="http://gov.cn/p2", summary="深圳创新"),  # 更短
    ]

    deduped = Orchestrator.deduplicate(policies)

    print(f"去重前: {len(policies)} 条")
    print(f"去重后: {len(deduped)} 条")
    for p in deduped:
        print(f"  - {p.title} | {p.url} | 摘要长度={len(p.summary)} | PDF={p.pdf_url}")

    assert len(deduped) == 2, f"应该去重为 2 条，实际 {len(deduped)}"
    # 验证保留了更长的摘要
    p1 = [p for p in deduped if "人才" in p.title][0]
    assert "更长更详细" in p1.summary, "应该保留摘要更长的版本"
    assert p1.pdf_url, "应该补充 PDF 链接"

    print(f"\n🎉 去重测试通过!")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Orchestrator 测试")
    parser.add_argument(
        "mode",
        nargs="?",
        default="plan",
        choices=["plan", "search", "full", "dedup", "all"],
        help="测试模式: plan=仅AI拆分, search=拆分+搜索, full=完整流程, dedup=去重测试",
    )
    parser.add_argument(
        "--company",
        default="光通信",
        choices=list(SAMPLE_COMPANIES.keys()),
        help="测试企业 (默认: 光通信)",
    )
    parser.add_argument("--budget", type=float, default=180, help="时间预算(秒), 默认180")
    parser.add_argument("--rounds", type=int, default=3, help="最大搜索轮次, 默认3")
    parser.add_argument("--delay", type=float, default=2.0, help="请求间隔(秒), 默认2.0")
    args = parser.parse_args()

    if args.mode == "dedup":
        test_dedup()
    elif args.mode == "plan":
        test_plan(args.company)
    elif args.mode == "search":
        await test_search(args.company, args.budget, args.rounds, args.delay)
    elif args.mode == "full":
        await test_full(args.company, args.budget, args.rounds, args.delay)
    elif args.mode == "all":
        test_dedup()
        test_plan(args.company)
        await test_search(args.company, args.budget, args.rounds, args.delay)


if __name__ == "__main__":
    asyncio.run(main())
