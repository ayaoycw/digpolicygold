"""
一次性测试脚本 — 在服务器上跑 browser-use 搜索微电子奖励政策
"""
import asyncio
import json
import logging
import sys
from datetime import datetime

# 配置日志 — 输出到终端 + 文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/test_run.log', mode='w', encoding='utf-8'),
    ]
)

from browser_use_worker import run_browser_task, PolicySearchResult

TASK = (
    "你的任务：找到上海市2024-2025年微电子（集成电路）行业的政府奖励政策。（百度已自动打开）\n\n"
    "【策略 — 直接从搜索开始】\n"
    "1. 在搜索框中输入: 上海 集成电路 奖励政策 2025 site:gov.cn\n"
    "2. 直接点击搜索结果标题链接（不要用 find_elements 找 href，百度的真实 URL 藏在跳转里）\n"
    "3. 进入政策页后用 extract 提取标题、来源、日期、摘要\n"
    "4. 在页面上找「文件下载」或 PDF 链接，点击获取 PDF URL\n"
    "5. 返回搜索结果页，点击下一个结果获取更多政策\n"
    "6. 尽可能收集多条政策，但至少1条含具体 URL/PDF 即可结束\n\n"
    "【规则】\n"
    "- 遇到验证码/拦截 → 立即 go_back\n"
    "- 不访问 qichacha/tianyancha/aiqicha 等\n"
    "- 每条政策必须提取: 标题、来源、URL、PDF链接、摘要、日期\n"
    "- 最终返回合法 JSON\n\n"
    "返回 JSON 格式：\n"
    '{"search_query": "实际搜索词", "target_industry": "集成电路/微电子", '
    '"target_region": "上海", '
    '"policies": [{"policy_title": "标题", "source": "发布机构", '
    '"url": "原文链接", "pdf_url": "PDF链接", "summary": "摘要", '
    '"publish_date": "日期", "applicable_industry": "适用行业", '
    '"key_support": "奖励/支持内容"}], '
    '"search_notes": "搜索过程备注"}'
)


async def main():
    print("=" * 60)
    print(f"🚀 测试开始: {datetime.now().strftime('%H:%M:%S')}")
    print(f"📝 任务: 上海 微电子/集成电路 奖励政策")
    print("=" * 60)

    result = await run_browser_task(
        TASK,
        output_model=PolicySearchResult,
        max_steps=15,
        use_vision="auto",
        headless=True,
    )

    print("\n" + "=" * 60)
    print(f"{'✅ 成功' if result['success'] else '❌ 失败'}")
    print(f"⏱  耗时: {result['duration']}s | 步数: {result['steps']}")

    if result.get("error"):
        print(f"❌ 错误: {result['error']}")

    if result.get("urls"):
        unique = list(dict.fromkeys(u for u in result["urls"] if u))
        print(f"\n📎 访问过的 URL ({len(unique)} 个):")
        for url in unique:
            print(f"   {url}")

    print(f"\n📄 最终结果:")
    print("-" * 60)
    r = result.get("structured") or result.get("result")
    if isinstance(r, dict):
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif r:
        print(str(r)[:3000])
    else:
        print("(无结果)")

    if result.get("parse_error"):
        print(f"\n⚠️  解析错误: {result['parse_error']}")

    print("=" * 60)

    # 保存结果
    with open('/tmp/test_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print("💾 结果已保存: /tmp/test_result.json")


if __name__ == "__main__":
    asyncio.run(main())
