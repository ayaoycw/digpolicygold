"""
Browser Use SDK 测试脚本（v0.11.9）
===================================
直接调用 browser-use 0.11.9 SDK，输入任务描述，输出结构化结果。
独立运行环境：/opt/browser-sdk/（与 web-ui 完全解耦）

用法：
    python test_browser_use.py                          # 交互模式，手动输入任务
    python test_browser_use.py "搜索北京今天天气"         # 命令行传入任务
    python test_browser_use.py --example links          # 运行预设示例

环境变量（从 .env 加载）：
    AZURE_OPENAI_ENDPOINT    — Azure OpenAI 端点
    AZURE_OPENAI_API_KEY     — Azure OpenAI API Key
"""

import asyncio
import json
import sys
import os
from typing import Optional
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# ─────────────────────────────────────────────
# 结构化输出模型（可选，按需使用）
# ─────────────────────────────────────────────

class SearchResult(BaseModel):
    """搜索结果 — 包含链接列表"""
    title: str
    url: str
    snippet: str = ""

class SearchResults(BaseModel):
    """多条搜索结果"""
    query: str
    results: list[SearchResult]

class FileLink(BaseModel):
    """文件/PDF 链接"""
    title: str
    url: str
    file_type: str = ""  # pdf, doc, etc.

class FileLinks(BaseModel):
    """多个文件链接"""
    query: str
    links: list[FileLink]

class ExtractedContent(BaseModel):
    """通用提取内容"""
    title: str
    content: str
    source_url: str = ""

class PolicyInfo(BaseModel):
    """奖励政策信息"""
    policy_title: str
    source: str = ""       # 来源：政府网站 / 微信公众号 等
    url: str = ""           # 原文链接
    pdf_url: str = ""       # PDF 链接（如有）
    summary: str = ""       # 政策摘要
    publish_date: str = ""  # 发布日期
    applicable: str = ""    # 适用说明

class CompanyPolicyResult(BaseModel):
    """公司信息 + 相关政策调研结果"""
    company_name: str
    registered_address: str = ""
    industry: str = ""
    company_type: str = ""
    established_date: str = ""
    company_summary: str = ""
    policies: list[PolicyInfo] = []
    search_notes: str = ""  # 搜索过程备注


# ─────────────────────────────────────────────
# 核心：调用 browser-use
# ─────────────────────────────────────────────

async def run_browser_task(
    task: str,
    output_model=None,
    max_steps: int = 20,
) -> dict:
    """
    执行 browser-use 任务（适配 0.11.9 API）。
    
    参数：
        task:          任务描述（自然语言，像写 prompt 一样）
        output_model:  Pydantic 模型类，要求 agent 返回结构化 JSON（可选）
        max_steps:     最大步数
    
    返回：
        {
            "result": "最终结果文本 或 结构化JSON",
            "urls": ["访问过的URL列表"],
            "steps": 步数,
            "duration": 耗时秒数,
            "extracted": ["每步提取的内容"],
            "success": True/False
        }
    """
    from browser_use import Agent, Browser, ChatAzureOpenAI

    # Azure OpenAI — 使用 browser-use 0.11.9 内置的 ChatAzureOpenAI
    llm = ChatAzureOpenAI(
        model=os.getenv("AZURE_OPENAI_MODEL", "o3"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
    )

    # 浏览器配置 — 0.11.9 直接传参给 Browser()
    # 使用系统安装的 google-chrome-stable，避免 Playwright CDN 被墙的问题
    # enable_default_extensions=False 避免在中国网络下载扩展超时
    browser = Browser(
        headless=True,
        executable_path="/usr/bin/google-chrome-stable",
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--lang=zh-CN",
            "--disable-gpu",
        ],
        chromium_sandbox=False,
        enable_default_extensions=False,
    )

    # 系统提示：中国网络环境适配
    system_prompt = (
        "你在中国大陆网络环境运行。"
        "搜索时使用百度(baidu.com)或搜狗(sogou.com)，不要使用 Google。"
        "所有输出使用中文。"
    )

    # 构建 Agent — 0.11.9 支持 output_model_schema
    agent_kwargs = dict(
        task=task,
        llm=llm,
        browser=browser,
        max_failures=5,
        extend_system_message=system_prompt,
        use_vision=False,  # 禁用视觉（截图），服务器内存有限+避免超时
    )
    if output_model:
        agent_kwargs["output_model_schema"] = output_model

    agent = Agent(**agent_kwargs)

    # 执行
    start = datetime.now()
    history = await agent.run(max_steps=max_steps)
    duration = (datetime.now() - start).total_seconds()

    # 提取结果
    final_text = history.final_result()
    
    result = {
        "success": final_text is not None,
        "result": final_text,
        "urls": history.urls() if hasattr(history, 'urls') else [],
        "steps": history.number_of_steps() if hasattr(history, 'number_of_steps') else 0,
        "duration": round(duration, 1),
        "extracted": history.extracted_content() if hasattr(history, 'extracted_content') else [],
    }

    # 如果有结构化输出模型，尝试解析
    if output_model and final_text:
        try:
            parsed = output_model.model_validate_json(final_text)
            result["structured"] = parsed.model_dump()
            result["result"] = parsed.model_dump()  # 替换为结构化数据
        except Exception as e:
            result["parse_error"] = str(e)

    # 关闭浏览器
    try:
        await browser.close()
    except:
        pass

    return result


# ─────────────────────────────────────────────
# 预设示例任务
# ─────────────────────────────────────────────

EXAMPLES = {
    "weather": {
        "task": '打开百度搜索"北京今天天气"，提取当前温度、天气状况、风力信息。',
        "model": None,
    },
    "links": {
        "task": (
            '在百度搜索"Python 异步编程教程"，'
            "找到前5个搜索结果的标题和链接。"
            "请以 JSON 格式返回，包含 query 和 results 列表，"
            "每个 result 包含 title、url、snippet 字段。"
        ),
        "model": SearchResults,
    },
    "pdf": {
        "task": (
            '在百度搜索"机器学习入门 PDF filetype:pdf"，'
            "找到前5个 PDF 文件的下载链接。"
            "请以 JSON 格式返回，包含 query 和 links 列表，"
            "每个 link 包含 title、url、file_type 字段。"
        ),
        "model": FileLinks,
    },
    "extract": {
        "task": (
            "打开 https://news.ycombinator.com/ ，"
            "提取排名前3的帖子标题和链接。"
        ),
        "model": None,
    },
    "company_policy": {
        "task": (
            '你的任务是调研"上海聿凡领光通信有限公司"的基本信息和可能适用的政府奖励政策。\n\n'
            "重要提示：\n"
            "- 企查查(qichacha.com)和天眼查(tianyancha.com)会拦截爬虫，不要试图进入，直接跳过\n"
            "- 优先使用爱企查(aiqicha.com)或百度百科，或者直接从百度搜索结果摘要中提取公司信息\n"
            "- 每次只操作一个标签页，避免打开过多标签\n"
            "- 如果一个页面加载缓慢或被拦截，立即返回搜索结果页尝试其他链接\n\n"
            "第一步：查找公司基本信息（控制在5步以内）\n"
            '1. 打开百度搜索"上海聿凡领光通信有限公司"，从搜索结果摘要中提取公司基本信息\n'
            "2. 如果摘要不够，尝试进入爱企查(aiqicha.com)的页面获取更多信息\n"
            "3. 记录：注册地址（特别是所在区）、行业、企业类型、成立日期\n\n"
            "第二步：搜索政府奖励政策（控制在10步以内）\n"
            '4. 在百度搜索"上海 [公司所在区] 光通信 产业扶持政策"或"上海 [公司所在区] 科技企业 奖励补贴"\n'
            "5. 查看搜索结果，优先点击政府网站(.gov.cn)的链接\n"
            "6. 提取政策标题、原文URL、PDF链接（如页面中有PDF下载按钮）\n\n"
            "第三步：搜狗微信搜索政策文章（控制在5步以内）\n"
            "7. 打开 weixin.sogou.com\n"
            '8. 搜索"上海 光通信 扶持政策"或"上海 通信产业 补贴奖励"\n'
            "9. 提取前3篇相关文章的标题和链接\n\n"
            "第四步：汇总为JSON返回\n"
            "务必返回合法的JSON字符串，包含以下字段：\n"
            '{"company_name": "公司名", "registered_address": "地址", "industry": "行业", '
            '"company_type": "类型", "established_date": "日期", "company_summary": "简介", '
            '"policies": [{"policy_title": "标题", "source": "来源", "url": "链接", '
            '"pdf_url": "PDF链接", "summary": "摘要", "publish_date": "日期", "applicable": "说明"}], '
            '"search_notes": "备注"}'
        ),
        "model": CompanyPolicyResult,
        "max_steps": 30,
    },
}


# ─────────────────────────────────────────────
# 输出格式化
# ─────────────────────────────────────────────

def print_result(result: dict):
    """美化输出结果"""
    print("\n" + "=" * 60)
    print(f"✅ 成功" if result["success"] else "❌ 失败")
    print(f"⏱  耗时: {result['duration']}s | 步数: {result['steps']}")
    
    if result.get("urls"):
        print(f"\n📎 访问过的 URL:")
        for url in result["urls"]:
            if url:
                print(f"   {url}")

    print(f"\n📄 最终结果:")
    print("-" * 60)
    
    r = result["result"]
    if isinstance(r, dict):
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif r:
        print(r)
    else:
        print("(无结果)")
    
    if result.get("parse_error"):
        print(f"\n⚠️  结构化解析失败: {result['parse_error']}")
    
    print("=" * 60)


def save_result(result: dict, filename: str = None):
    """保存结果到文件"""
    if not filename:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"result_{ts}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n💾 结果已保存到: {filename}")


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────

async def main():
    # 解析命令行参数
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        # 预设示例
        if arg == "--example" and len(sys.argv) > 2:
            example_name = sys.argv[2]
            if example_name not in EXAMPLES:
                print(f"可用示例: {', '.join(EXAMPLES.keys())}")
                return
            ex = EXAMPLES[example_name]
            steps = ex.get("max_steps", 20)
            print(f"🚀 运行示例: {example_name}")
            print(f"📝 任务: {ex['task'][:80]}...")
            print(f"📊 最大步数: {steps}")
            result = await run_browser_task(ex["task"], output_model=ex["model"], max_steps=steps)
            print_result(result)
            save_result(result)
            return
        
        # 直接传入任务
        if arg == "--help":
            print(__doc__)
            print(f"\n预设示例: {', '.join(EXAMPLES.keys())}")
            print("  python test_browser_use.py --example links")
            return
        
        task = " ".join(sys.argv[1:])
        print(f"🚀 执行任务: {task}")
        result = await run_browser_task(task)
        print_result(result)
        save_result(result)
        return

    # 交互模式
    print("=" * 60)
    print("🌐 Browser Use SDK 测试工具")
    print("=" * 60)
    print()
    print("💡 怎么写任务（像写 prompt）：")
    print("   1. 具体明确 — 说清楚去哪个网站、做什么、返回什么")
    print("   2. 分步骤写 — 用 1. 2. 3. 编号更可靠")
    print("   3. 指定格式 — 要 JSON/链接列表，直接在任务里说")
    print()
    print("📌 示例任务：")
    print('   "在百度搜索今天北京天气，返回温度和天气状况"')
    print('   "打开 zhihu.com 找到热搜前3个问题的标题和链接"')
    print('   "搜索 Python 异步教程 PDF，找到3个下载链接"')
    print()
    print("⌨️  输入 'q' 退出 | 'examples' 查看预设示例")
    print()

    while True:
        task = input("📝 输入任务> ").strip()
        
        if not task:
            continue
        if task.lower() == "q":
            break
        if task.lower() == "examples":
            for name, ex in EXAMPLES.items():
                print(f"  {name:10s} — {ex['task'][:60]}...")
            print(f"\n运行示例: 输入 'run <名称>'，如 'run links'")
            continue
        if task.lower().startswith("run "):
            name = task[4:].strip()
            if name in EXAMPLES:
                ex = EXAMPLES[name]
                steps = ex.get("max_steps", 20)
                print(f"🚀 运行示例: {name}")
                result = await run_browser_task(ex["task"], output_model=ex["model"], max_steps=steps)
                print_result(result)
                save_result(result)
            else:
                print(f"未知示例: {name}，可用: {', '.join(EXAMPLES.keys())}")
            continue

        # 执行用户输入的任务
        print(f"🚀 执行中...")
        result = await run_browser_task(task)
        print_result(result)
        
        save = input("💾 保存结果? (y/N) ").strip().lower()
        if save == "y":
            save_result(result)


if __name__ == "__main__":
    asyncio.run(main())
