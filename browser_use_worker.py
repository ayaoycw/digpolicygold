"""
Browser Use Worker v0.13
====================================
基于 browser-use 0.11.9 SDK，作为 Worker 负责复杂、困难的搜索与信息提取任务。

实现 BaseWorker 接口，search() 返回统一的 WorkerResult。

定位：
    - 快速搜索 → 用 WebSearchWorker（web_search_worker.py）
    - 深度搜索 → 用本 Worker（browser_use_worker.py）
      例如：需要多步骤翻页、反爬绕过、PDF 提取、政府网站深度采集等场景

用法：
    # 作为 Worker 使用（推荐）
    from browser_use_worker import BrowserUseWorker
    worker = BrowserUseWorker()
    result = worker.search("上海光通信产业扶持政策")  # → WorkerResult

    # 直接调用低级 API
    from browser_use_worker import run_browser_task
    result = await run_browser_task("任务描述")

环境变量（从 .env 加载）：
    AZURE_OPENAI_ENDPOINT    — Azure OpenAI 端点
    AZURE_OPENAI_API_KEY     — Azure OpenAI API Key
"""

import asyncio
import json
import re
import sys
import os
from typing import Optional
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

from models import BaseWorker, WorkerResult, PolicyItem


# ─────────────────────────────────────────────
# 结构化输出模型
# ─────────────────────────────────────────────

class SearchResult(BaseModel):
    """搜索结果"""
    title: str
    url: str
    snippet: str = ""

class SearchResults(BaseModel):
    """多条搜索结果"""
    query: str
    results: list[SearchResult]

class PolicyInfo(BaseModel):
    """单条政策信息"""
    policy_title: str
    source: str = ""
    url: str = ""
    pdf_url: str = ""
    summary: str = ""
    publish_date: str = ""
    applicable_industry: str = ""
    key_support: str = ""       # 支持内容摘要（资金额度、补贴比例等）

class PolicySearchResult(BaseModel):
    """政策搜索结果"""
    search_query: str = ""
    target_industry: str = ""
    target_region: str = ""
    policies: list[PolicyInfo] = []
    search_notes: str = ""

class PageContent(BaseModel):
    """页面内容提取"""
    title: str = ""
    url: str = ""
    main_content: str = ""
    pdf_links: list[str] = []
    related_links: list[str] = []


# ─────────────────────────────────────────────
# 下载目录
# ─────────────────────────────────────────────
DOWNLOAD_DIR = "/tmp/downloads"


# ─────────────────────────────────────────────
# 核心：创建 LLM 和 Browser
# ─────────────────────────────────────────────

def _create_azure_llm(model: str):
    """创建 Azure OpenAI LLM 实例（通用工厂）"""
    from browser_use import ChatAzureOpenAI
    return ChatAzureOpenAI(
        model=model,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
    )


def create_llm():
    """主力 LLM：o3（强推理，贵）"""
    return _create_azure_llm(os.getenv("AZURE_OPENAI_MODEL", "o3"))


def create_fallback_llm():
    """
    备用 LLM：o4-mini（便宜、快，推理能力够用）
    
    触发条件：主力 o3 耗尽重试后仍失败（429/500/502/503/504）
    一旦切换，剩余步骤全部使用 fallback。
    """
    return _create_azure_llm(os.getenv("AZURE_OPENAI_FALLBACK_MODEL", "o4-mini"))


def create_extraction_llm():
    """
    页面提取专用 LLM：gpt-4o（便宜、快，只需提取文本）
    
    用于 extract action 的内容提取，不需要推理能力，省 token 费用。
    o3 提取一次页面 ~5000 token，gpt-4o 只需 ~500 token。
    """
    return _create_azure_llm(os.getenv("AZURE_OPENAI_EXTRACT_MODEL", "gpt-4o"))


def create_browser_profile(headless: bool = True):
    """
    创建浏览器配置（BrowserProfile）。
    
    关键设置：
    - 视觉友好的窗口尺寸（1920x1080）
    - PDF 自动下载
    - 禁用安全限制以访问各类网站
    - 中文语言环境
    """
    from browser_use.browser.profile import BrowserProfile

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    profile = BrowserProfile(
        # 浏览器可执行文件 — 使用系统 Chrome
        executable_path="/usr/bin/google-chrome-stable",
        headless=headless,
        
        # 启动参数
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--lang=zh-CN",
            "--disable-gpu",
            "--disable-blink-features=AutomationControlled",  # 隐藏自动化特征
            "--disable-features=IsolateOrigins,site-per-process",  # 允许跨域iframe
            f"--window-size=1920,1080",
        ],
        chromium_sandbox=False,
        enable_default_extensions=False,
        
        # PDF — 不要自动下载（会触发 DownloadsWatchdog 超时）
        # agent 可以通过 navigate 到 PDF URL 来读取内容
        auto_download_pdfs=False,
        downloads_path=DOWNLOAD_DIR,
        
        # 安全 — 放宽限制，能访问更多网站
        disable_security=True,
        
        # 页面等待
        minimum_wait_page_load_time=0.5,
        wait_for_network_idle_page_load_time=1.0,
        wait_between_actions=0.3,
        
        # DOM 高亮（帮助视觉模式理解交互元素）
        highlight_elements=True,
    )
    return profile


# ─────────────────────────────────────────────
# 系统提示模板
# ─────────────────────────────────────────────

SYSTEM_PROMPT_CN = """你在中国大陆网络环境运行。请严格遵守以下规则：

【搜索引擎】
- 使用百度(baidu.com)搜索，不要使用 Google
- 百度已自动打开，直接在搜索框中输入关键词即可
- 不要直接在URL栏构造搜索URL
- site:gov.cn 是百度搜索语法，不是网站地址！不要尝试直接访问 gov.cn

【反爬策略 — 最重要】
- 如果遇到验证码页面（看到滑块、图片验证、"请验证您是人类"等），立即执行 go_back 返回上一页，换一个链接或换一种搜索词
- 绝对不要在验证码页面上反复尝试，浪费步数。一旦看到验证码，马上离开
- 以下网站会拦截爬虫，直接跳过不要尝试进入：
  * 企查查 qichacha.com
  * 天眼查 tianyancha.com
  * 搜狗微信 weixin.sogou.com
  * 爱企查 aiqicha.com（百度验证码保护）
- 优先访问政府官网(.gov.cn)，这些网站通常不会拦截

【高效浏览】
- 每次只操作一个标签页
- 如果页面3秒内无响应，执行 go_back 换下一个
- 从搜索结果摘要中就能提取到的信息，不需要点进去
- 政府网站(.gov.cn)优先级最高，直接进入提取全文
- 提取页面内容时，使用 extract 而不是反复 find_elements
- 百度搜索结果的链接是跳转 URL，不是直接的 .gov.cn 链接。直接点击标题文本即可，不要用 find_elements 找 href 匹配 .gov.cn

【输出】
- 所有输出使用中文
- 返回合法的 JSON 字符串
"""


# ─────────────────────────────────────────────
# 核心：运行 browser-use 任务
# ─────────────────────────────────────────────

async def run_browser_task(
    task: str,
    output_model=None,
    max_steps: int = 20,
    use_vision: str = "auto",
    headless: bool = True,
    system_prompt: str = None,
) -> dict:
    """
    执行 browser-use 任务。

    参数：
        task:          任务描述（自然语言）
        output_model:  Pydantic 模型类（可选，返回结构化 JSON）
        max_steps:     最大步数
        use_vision:    视觉模式 — "auto"（SDK自动决定）| True（每步截图）| False（关闭）
        headless:      无头模式，默认 True
        system_prompt: 自定义系统提示（默认使用中国网络适配提示）

    返回：
        {
            "result": 最终结果（文本或结构化dict）,
            "urls": ["访问过的URL"],
            "steps": 步数,
            "duration": 耗时秒数,
            "extracted": ["每步提取的内容"],
            "success": True/False,
            "downloads": ["下载的文件路径"],
        }
    """
    from browser_use import Agent
    from browser_use.browser.session import BrowserSession

    llm = create_llm()
    fallback = create_fallback_llm()
    extraction = create_extraction_llm()
    profile = create_browser_profile(headless=headless)
    browser_session = BrowserSession(browser_profile=profile)

    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT_CN

    # 构建 Agent
    agent_kwargs = dict(
        task=task,
        llm=llm,
        browser_session=browser_session,
        max_failures=5,
        extend_system_message=system_prompt,
        use_vision=use_vision,
        
        # 备用 LLM — o3 挂了自动切到 o4-mini（429/500/502/503/504）
        fallback_llm=fallback,
        
        # 页面提取专用 LLM — gpt-4o（便宜、快，只做文本提取）
        page_extraction_llm=extraction,
        
        # 视觉优化 — auto 模式下 SDK 自行决定何时截图；截图时使用低分辨率省 token
        vision_detail_level="low",
        
        # 关闭 judge — 避免 judge verdict 污染 final_result
        use_judge=False,
        
        # 规划 — 遇到停滞时重新规划
        enable_planning=True,
        planning_replan_on_stall=2,
        
        # 循环检测 — 更积极地跳出循环
        loop_detection_enabled=True,
        loop_detection_window=10,
        
        # 步骤超时
        step_timeout=120,
        
        # 文件系统路径（PDF下载目录）
        file_system_path=DOWNLOAD_DIR,
        
        # 每步最多执行的动作数（官方默认 4，允许 agent 一步完成多个操作）
        max_actions_per_step=4,
        
        # 预操作 — 直接打开百度，省去 LLM "打开百度" 的步骤（节省 1-2 步 + token）
        initial_actions=[
            {'navigate': {'url': 'https://www.baidu.com'}},
        ],
    )
    if output_model:
        agent_kwargs["output_model_schema"] = output_model

    agent = Agent(**agent_kwargs)

    # 执行
    start = datetime.now()
    try:
        history = await agent.run(max_steps=max_steps)
    except Exception as e:
        duration = (datetime.now() - start).total_seconds()
        return {
            "success": False,
            "result": None,
            "error": str(e),
            "urls": [],
            "steps": 0,
            "duration": round(duration, 1),
            "extracted": [],
            "downloads": [],
        }
    duration = (datetime.now() - start).total_seconds()

    # 提取结果
    final_text = history.final_result()

    result = {
        "success": history.is_done() and final_text is not None and len(str(final_text).strip()) > 0,
        "result": final_text,
        "urls": history.urls(),
        "steps": history.number_of_steps(),
        "duration": round(duration, 1),
        "extracted": history.extracted_content(),
        "downloads": _list_downloads(),
    }

    # 结构化解析 — 优先使用 SDK 内置的 history.structured_output
    if output_model:
        parsed = None

        # 方式1: SDK 内置 structured_output（自动解析，无需手动清理 JSON）
        try:
            structured = history.structured_output
            if structured is not None:
                parsed = structured
        except Exception:
            pass

        # 方式2: 回退到手动解析（兼容 structured_output 失败的情况）
        if parsed is None and final_text:
            cleaned = _clean_final_result(str(final_text))
            try:
                parsed = output_model.model_validate_json(cleaned)
            except Exception:
                json_text = _extract_json(str(final_text))
                if json_text:
                    try:
                        parsed = output_model.model_validate_json(json_text)
                    except Exception as e:
                        result["parse_error"] = str(e)

        if parsed is not None:
            dumped = parsed.model_dump() if hasattr(parsed, 'model_dump') else parsed
            result["structured"] = dumped
            result["result"] = dumped

    # 关闭浏览器
    try:
        await browser_session.close()
    except:
        pass

    return result


# ─────────────────────────────────────────────
# BrowserUseWorker — 实现 BaseWorker 接口
# ─────────────────────────────────────────────

class BrowserUseWorker(BaseWorker):
    """
    基于 browser-use 无头浏览器的深度搜索 Worker。

    实现 BaseWorker 接口:
        worker = BrowserUseWorker()
        result = worker.search("浦东新区光通信产业扶持政策")  # → WorkerResult
    """

    name = "browser_use"

    def __init__(
        self,
        max_steps: int = 25,
        use_vision: str = "auto",
        headless: bool = True,
    ):
        self.max_steps = max_steps
        self.use_vision = use_vision
        self.headless = headless

    def _build_task(self, query: str) -> str:
        """根据查询构建 browser-use 任务指令"""
        return (
            f"你的任务：搜索并提取以下查询相关的政策信息：{query}\n\n"
            "【策略】（百度已自动打开，直接从搜索开始）\n"
            f"1. 在搜索框中输入: {query} site:gov.cn\n"
            "2. 从搜索结果页直接点击结果标题链接（不要用 find_elements 找 href，百度会隐藏真实 URL）\n"
            "3. 进入政策页后用 extract 提取详情，找 PDF 链接\n"
            "4. 尽可能收集多条政策，但至少1条即可结束\n\n"
            "【规则】\n"
            "- 遇到验证码/拦截 → 立即 go_back\n"
            "- 不访问 qichacha/tianyancha/aiqicha 等\n"
            "- 每条政策提取: 标题、来源、URL、PDF链接、摘要、日期\n\n"
            "返回 JSON：\n"
            '{"policies": [{"policy_title": "标题", "source": "机构", '
            '"url": "链接", "pdf_url": "PDF链接", "summary": "摘要", '
            '"publish_date": "日期", "applicable_industry": "行业", '
            '"key_support": "扶持内容"}]}'
        )

    def search(self, query: str, **kwargs) -> WorkerResult:
        """
        执行深度搜索，返回统一的 WorkerResult（实现 BaseWorker 接口）

        内部通过 asyncio 调用 run_browser_task()。
        """
        import time as _time
        start = _time.time()

        task = kwargs.get("task") or self._build_task(query)
        max_steps = kwargs.get("max_steps", self.max_steps)

        try:
            # 运行异步任务
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 已在事件循环中 — 需要在线程池中运行
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    raw = pool.submit(
                        asyncio.run,
                        run_browser_task(
                            task,
                            output_model=PolicySearchResult,
                            max_steps=max_steps,
                            use_vision=self.use_vision,
                            headless=self.headless,
                        )
                    ).result()
            else:
                raw = asyncio.run(
                    run_browser_task(
                        task,
                        output_model=PolicySearchResult,
                        max_steps=max_steps,
                        use_vision=self.use_vision,
                        headless=self.headless,
                    )
                )

            elapsed = round(_time.time() - start, 1)
            return self._raw_to_worker_result(query, raw, elapsed)

        except Exception as e:
            elapsed = round(_time.time() - start, 1)
            return WorkerResult(
                query=query,
                worker=self.name,
                duration=elapsed,
                error=str(e),
            )

    async def search_async(self, query: str, **kwargs) -> WorkerResult:
        """异步版本的 search（server.py 中使用）"""
        import time as _time
        start = _time.time()

        task = kwargs.get("task") or self._build_task(query)
        max_steps = kwargs.get("max_steps", self.max_steps)

        try:
            raw = await run_browser_task(
                task,
                output_model=PolicySearchResult,
                max_steps=max_steps,
                use_vision=self.use_vision,
                headless=self.headless,
            )
            elapsed = round(_time.time() - start, 1)
            return self._raw_to_worker_result(query, raw, elapsed)
        except Exception as e:
            elapsed = round(_time.time() - start, 1)
            return WorkerResult(
                query=query,
                worker=self.name,
                duration=elapsed,
                error=str(e),
            )

    def _raw_to_worker_result(self, query: str, raw: dict, elapsed: float) -> WorkerResult:
        """把 run_browser_task 的原始 dict 转为 WorkerResult"""
        policies = []

        # 从 structured 提取
        structured = raw.get("structured") or raw.get("result")
        if isinstance(structured, dict):
            for p in structured.get("policies", []):
                policies.append(PolicyItem(
                    title=p.get("policy_title", ""),
                    url=p.get("url", ""),
                    source=p.get("source", ""),
                    date=p.get("publish_date", ""),
                    summary=p.get("summary", ""),
                    support=p.get("key_support", ""),
                    pdf_url=p.get("pdf_url", ""),
                    industry=p.get("applicable_industry", ""),
                ))

        return WorkerResult(
            query=query,
            policies=policies,
            sources=raw.get("urls", []),
            worker=self.name,
            duration=elapsed,
            error=None if raw.get("success") else raw.get("error"),
            raw_answer=json.dumps(structured, ensure_ascii=False) if isinstance(structured, dict) else str(raw.get("result", "")),
        )


def _clean_final_result(text: str) -> str:
    """清理 final_result 中可能被附加的 judge verdict 等非 JSON 内容"""
    if not text:
        return text
    
    # 如果文本以 { 开头，尝试找到匹配的 }
    text = text.strip()
    if text.startswith("{"):
        depth = 0
        for i, ch in enumerate(text):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[:i+1]
    
    # 如果文本以 [ 开头，找匹配的 ]
    if text.startswith("["):
        depth = 0
        for i, ch in enumerate(text):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[:i+1]
    
    return text


def _extract_json(text: str) -> Optional[str]:
    """从混合文本中提取第一个完整的 JSON 对象"""
    # 找第一个 { 和最后一个 }
    start = text.find("{")
    if start == -1:
        return None
    
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    json.loads(candidate)
                    return candidate
                except:
                    continue
    return None


def _list_downloads() -> list[str]:
    """列出下载目录中的文件"""
    if not os.path.exists(DOWNLOAD_DIR):
        return []
    files = []
    for f in os.listdir(DOWNLOAD_DIR):
        fpath = os.path.join(DOWNLOAD_DIR, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            files.append(f"{f} ({size} bytes)")
    return files


# ─────────────────────────────────────────────
# 预设示例任务（focused, single-objective）
# ─────────────────────────────────────────────

EXAMPLES = {
    "weather": {
        "task": '百度已打开，直接在搜索框输入“北京今天天气”，提取当前温度、天气状况、风力信息。',
        "model": None,
        "max_steps": 10,
    },

    "policy_search": {
        "task": (
            "你的任务：找到2025-2026年上海市适合「光通信」行业的政府产业扶持政策。\n\n"
            "【策略 — 百度已自动打开，按顺序执行】\n\n"
            "第一轮搜索（步骤1-5）：百度搜索政府政策\n"
            "1. 在搜索框中输入: 上海 光通信 产业扶持政策 2025 site:gov.cn\n"
            "2. 查看搜索结果，从摘要中提取信息（标题、日期、来源）\n"
            "3. 点击第1个 .gov.cn 链接，进入后提取政策详情和PDF链接\n"
            "4. 返回搜索结果页（go_back），点击第2个 .gov.cn 链接获取第2条政策\n\n"
            "第二轮搜索（步骤5-8）：换搜索词扩大范围\n"
            "5. 新搜索: 上海 通信产业 补贴 奖励政策 site:gov.cn\n"
            "6. 从结果中再获取1-2条不同的政策\n\n"
            "第三轮（步骤7-10）：直接访问政府网站\n"
            "7. 打开 https://sheitc.sh.gov.cn （上海市经信委）\n"
            "8. 在网站上搜索「光通信」或「通信产业」\n"
            "9. 提取相关政策信息\n\n"
            "【重要规则】\n"
            "- 遇到验证码/拦截 → 立即 go_back，换下一个链接\n"
            "- 不要访问 qichacha/tianyancha/aiqicha/weixin.sogou 等会拦截的网站\n"
            "- 每个政策都要提取: 标题、来源、URL、PDF链接、摘要、发布日期\n"
            "- 至少收集2条政策信息后才能结束\n\n"
            "返回 JSON 格式（必须是合法 JSON）：\n"
            '{"search_query": "实际搜索词", "target_industry": "光通信", '
            '"target_region": "上海", '
            '"policies": [{"policy_title": "标题", "source": "发布机构", '
            '"url": "原文链接", "pdf_url": "PDF链接", "summary": "摘要", '
            '"publish_date": "日期", "applicable_industry": "适用行业", '
            '"key_support": "支持内容"}], '
            '"search_notes": "搜索过程备注"}'
        ),
        "model": PolicySearchResult,
        "max_steps": 25,
    },

    "gov_direct": {
        "task": (
            "你的任务：直接访问上海市浦东新区政府网站，查找通信产业相关的扶持政策。\n\n"
            "步骤：\n"
            "1. 直接打开 https://www.pudong.gov.cn/\n"
            "2. 在网站的搜索功能中搜索「通信 产业 扶持」或「光通信」\n"
            "3. 如果没有搜索框，尝试访问政策公开栏目\n"
            "4. 找到与通信产业相关的政策文件，提取标题、日期、URL、PDF链接\n"
            "5. 如果找到 PDF 下载按钮，点击下载\n\n"
            "返回 JSON 格式：\n"
            '{"search_query": "通信 产业 扶持", "target_industry": "光通信", '
            '"target_region": "上海浦东新区", '
            '"policies": [{"policy_title": "标题", "source": "来源", '
            '"url": "链接", "pdf_url": "PDF链接", "summary": "摘要", '
            '"publish_date": "日期", "applicable_industry": "适用行业", '
            '"key_support": "支持内容"}], '
            '"search_notes": "备注"}'
        ),
        "model": PolicySearchResult,
        "max_steps": 15,
    },

    "pdf_download": {
        "task": (
            "你的任务：下载一个政策PDF文件。\n\n"
            "步骤：\n"
            "1. 直接打开这个URL: https://www.pudong.gov.cn/zwgk/gfxwj-kjwzcwj/2024/299/333469.html\n"
            "2. 在页面中寻找 PDF 下载链接或按钮\n"
            "3. 点击下载 PDF 文件\n"
            "4. 提取页面中的政策标题和主要内容摘要\n\n"
            "返回页面标题、PDF链接地址、以及政策主要内容概要。"
        ),
        "model": None,
        "max_steps": 10,
    },

    "captcha_test": {
        "task": (
            "你的任务：测试反验证码策略。（百度已自动打开）\n\n"
            "步骤：\n"
            "1. 在搜索框输入「上海聿凡领光通信有限公司」\n"
            "2. 从搜索结果摘要中提取公司基本信息（地址、行业、成立日期等）\n"
            "   - 不要点击爱企查、企查查、天眼查的链接\n"
            "3. 如果搜索结果摘要中有足够信息，直接返回\n"
            "4. 如果需要更多信息，尝试点击其他链接（如百度百科）\n"
            "5. 遇到任何验证码/拦截页面，立即 go_back 返回\n\n"
            "返回公司的：名称、注册地址、行业、企业类型、成立日期。"
        ),
        "model": None,
        "max_steps": 10,
    },

    "extract_page": {
        "task": (
            "你的任务：提取指定网页的完整内容。\n\n"
            "步骤：\n"
            "1. 打开 https://www.pudong.gov.cn/zwgk/gfxwj-kjwzcwj/2024/299/333469.html\n"
            "2. 等待页面加载完成\n"
            "3. 提取页面标题\n"
            "4. 提取正文全部内容\n"
            "5. 找到页面中所有的 PDF 下载链接\n"
            "6. 找到页面底部的相关链接\n\n"
            "返回 JSON 格式：\n"
            '{"title": "标题", "url": "URL", "main_content": "正文内容", '
            '"pdf_links": ["PDF链接1"], "related_links": ["相关链接1"]}'
        ),
        "model": PageContent,
        "max_steps": 8,
    },
}


# ─────────────────────────────────────────────
# 输出格式化
# ─────────────────────────────────────────────

def print_result(result: dict):
    """美化输出结果"""
    print("\n" + "=" * 60)
    print(f"{'✅ 成功' if result['success'] else '❌ 失败'}")
    print(f"⏱  耗时: {result['duration']}s | 步数: {result['steps']}")

    if result.get("error"):
        print(f"\n❌ 错误: {result['error']}")

    if result.get("downloads"):
        print(f"\n📥 下载的文件:")
        for f in result["downloads"]:
            print(f"   {f}")

    if result.get("urls"):
        unique_urls = list(dict.fromkeys(u for u in result["urls"] if u))
        print(f"\n📎 访问过的 URL ({len(unique_urls)} 个):")
        for url in unique_urls[:10]:
            print(f"   {url}")
        if len(unique_urls) > 10:
            print(f"   ... 还有 {len(unique_urls)-10} 个")

    print(f"\n📄 最终结果:")
    print("-" * 60)

    r = result["result"]
    if isinstance(r, dict):
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif r:
        if len(r) > 3000:
            print(r[:3000])
            print(f"\n... (截断，共 {len(r)} 字符)")
        else:
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
            vision = ex.get("use_vision", True)
            print(f"🚀 运行示例: {example_name}")
            print(f"📝 任务: {ex['task'][:100]}...")
            print(f"📊 最大步数: {steps} | 视觉: {'开' if vision else '关'}")
            result = await run_browser_task(
                ex["task"],
                output_model=ex.get("model"),
                max_steps=steps,
                use_vision=vision,
            )
            print_result(result)
            save_result(result)
            return

        if arg == "--help":
            print(__doc__)
            print(f"\n预设示例: {', '.join(EXAMPLES.keys())}")
            print("  python browser_use_worker.py --example policy_search")
            return

        # 直接传入任务
        task = " ".join(sys.argv[1:])
        print(f"🚀 执行任务: {task}")
        result = await run_browser_task(task)
        print_result(result)
        save_result(result)
        return

    # 交互模式
    print("=" * 60)
    print("🌐 Browser Use 深度挖掘工具 v0.12")
    print("=" * 60)
    print()
    print("💡 怎么写任务（像写 prompt）：")
    print("   1. 具体明确 — 说清楚去哪个网站、做什么、返回什么")
    print("   2. 分步骤写 — 用 1. 2. 3. 编号更可靠")
    print("   3. 指定格式 — 要 JSON/链接列表，直接在任务里说")
    print()
    print("📌 预设示例：")
    for name, ex in EXAMPLES.items():
        print(f"   {name:15s} — {ex['task'][:50]}...")
    print()
    print("⌨️  'q' 退出 | 'run <示例名>' 运行示例")
    print()

    while True:
        task = input("📝 输入任务> ").strip()

        if not task:
            continue
        if task.lower() == "q":
            break
        if task.lower().startswith("run "):
            name = task[4:].strip()
            if name in EXAMPLES:
                ex = EXAMPLES[name]
                steps = ex.get("max_steps", 20)
                print(f"🚀 运行示例: {name}")
                result = await run_browser_task(
                    ex["task"], output_model=ex.get("model"), max_steps=steps
                )
                print_result(result)
                save_result(result)
            else:
                print(f"未知示例: {name}，可用: {', '.join(EXAMPLES.keys())}")
            continue

        print(f"🚀 执行中...")
        result = await run_browser_task(task)
        print_result(result)

        save = input("💾 保存结果? (y/N) ").strip().lower()
        if save == "y":
            save_result(result)


if __name__ == "__main__":
    asyncio.run(main())
