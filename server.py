"""
Policy Search API Server v0.14
==============================
FastAPI 后端，使用 Orchestrator 智能搜索：
  AI 分析企业信息 → 拆分多层搜索任务 → Web Search → AI 评审回路 → 去重汇总

通过 SSE 实时推送搜索过程和结果。

启动方式（阿里云服务器）：
    cd /opt/browser-sdk
    xvfb-run --auto-servernum python3 -u server.py

端口: 8000（nginx 反代 /api/ → 8000）
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.web_search")

# 统一模型
from models import WorkerResult

# Orchestrator 智能调度
from orchestrator import Orchestrator


# ─────────────────────────────────────────────
# 搜索日志持久化 — 每次搜索保存完整日志文件
# ─────────────────────────────────────────────

SEARCH_LOG_DIR = Path("/opt/browser-sdk/search_logs")
SEARCH_LOG_DIR.mkdir(parents=True, exist_ok=True)


def save_search_log(mode: str, query: str, log_lines: list[str], result: "WorkerResult"):
    """
    保存一次搜索的完整日志到文件。
    文件名: search_YYYYMMDD_HHMMSS_{mode}.log
    内容: 搜索参数 → 过程日志 → 结果摘要（含清晰的链接和PDF链接）
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = SEARCH_LOG_DIR / f"search_{ts}_{mode}.log"

    lines = []
    lines.append("=" * 70)
    lines.append(f"搜索日志 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"模式: {mode}")
    lines.append(f"查询: {query}")
    lines.append(f"Worker: {result.worker}")
    lines.append(f"耗时: {result.duration}s")
    lines.append(f"状态: {'✅ 成功' if result.success else '❌ 失败'}")
    if result.error:
        lines.append(f"错误: {result.error}")
    lines.append("=" * 70)

    # 过程日志
    lines.append("")
    lines.append("── 搜索过程 ──")
    for log_line in log_lines:
        lines.append(log_line)

    # Token 用量
    if result.token_usage:
        lines.append("")
        lines.append(f"── Token 用量 ──")
        lines.append(json.dumps(result.token_usage, ensure_ascii=False, indent=2))

    # 引用来源
    if result.sources:
        lines.append("")
        lines.append(f"── 引用来源 ({len(result.sources)} 个) ──")
        for i, url in enumerate(result.sources, 1):
            lines.append(f"  {i}. {url}")

    # 政策结果 — 清晰列出每条政策的链接和PDF
    lines.append("")
    lines.append(f"── 搜索结果: {result.policy_count} 条政策 ──")
    if result.policies:
        for i, p in enumerate(result.policies, 1):
            lines.append("")
            lines.append(f"  [{i}] {p.title}")
            lines.append(f"       来源: {p.source or '未知'}")
            lines.append(f"       日期: {p.date or '未知'}")
            lines.append(f"       行业: {p.industry or '未知'}")
            lines.append(f"       摘要: {p.summary or '无'}")
            lines.append(f"       扶持: {p.support or '无'}")
            lines.append(f"       🔗 原文链接: {p.url or '无'}")
            lines.append(f"       📥 PDF链接:  {p.pdf_url or '无'}")
    else:
        lines.append("  (无结果)")

    # LLM 原始回答
    if result.raw_answer:
        lines.append("")
        lines.append("── LLM 原始回答 ──")
        lines.append(result.raw_answer[:5000])

    lines.append("")
    lines.append("=" * 70)

    try:
        filename.write_text("\n".join(lines), encoding="utf-8")
        logging.getLogger(__name__).info(f"搜索日志已保存: {filename}")
    except Exception as e:
        logging.getLogger(__name__).error(f"保存搜索日志失败: {e}")

    return str(filename)


# ─────────────────────────────────────────────
# 日志捕获器 — 把 browser-use 日志推送到 SSE
# ─────────────────────────────────────────────

class LogCapture(logging.Handler):
    """捕获日志到 asyncio.Queue，供 SSE 流式推送"""

    # 忽略的 logger 名称前缀（太吵或无关）
    _IGNORE = {'uvicorn', 'httpx', 'httpcore', 'asyncio', 'watchfiles',
               'multipart', 'hpack', 'h2', 'charset_normalizer', 'PIL'}

    def __init__(self):
        super().__init__()
        self.queue: asyncio.Queue = asyncio.Queue()
        self._loop = None

    def set_loop(self, loop):
        self._loop = loop

    def emit(self, record):
        # 排除系统/网络库的噪音日志
        top = record.name.split('.')[0]
        if top in self._IGNORE:
            return
        msg = self.format(record)
        if msg.strip() and self._loop and self._loop.is_running():
            try:
                self._loop.call_soon_threadsafe(self.queue.put_nowait, msg)
            except Exception:
                pass


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(title="Policy Search API", version="0.14")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/policy-search/stream")
async def policy_search_stream(
    company_name: str = Query("", description="企业名称"),
    industry: str = Query(..., description="行业，如 光通信"),
    region: str = Query("上海", description="地区"),
    district: str = Query("", description="区，如 浦东新区"),
    tags: str = Query("", description="企业标签，逗号分隔"),
    registered_capital: str = Query("", description="注册资本"),
    employees: str = Query("", description="员工规模"),
    founded: str = Query("", description="成立时间"),
):
    """SSE 流式智能搜索 — Orchestrator 驱动"""

    # 构建企业信息 dict
    company_info = {
        "name": company_name or f"{region}{district} {industry}企业",
        "industry": industry,
        "region": f"{region} {district}".strip() if district else region,
        "tags": [t.strip() for t in tags.split(",") if t.strip()] if tags else [],
    }
    if registered_capital:
        company_info["registered_capital"] = registered_capital
    if employees:
        company_info["employees"] = employees
    if founded:
        company_info["founded"] = founded

    async def event_generator():
        log_lines = []
        log_queue: asyncio.Queue = asyncio.Queue()

        def on_log(msg: str):
            """orchestrator 的日志回调 → 放入队列"""
            try:
                log_queue.put_nowait(msg)
            except Exception:
                pass

        yield _sse({'type': 'status', 'message': '🧠 智能搜索启动中...'})

        orch = Orchestrator(
            on_log=on_log,
            time_budget=180.0,
            max_rounds=2,
            request_delay=3.0,
        )

        # 在后台线程运行 orchestrator
        result_holder = {}

        async def run_orch():
            result_holder['result'] = await orch.run(company_info, skip_browse_use=True)

        task = asyncio.ensure_future(run_orch())

        # 边执行边推送日志
        while not task.done():
            try:
                msg = await asyncio.wait_for(log_queue.get(), timeout=1.0)
                if msg.strip():
                    log_lines.append(msg.strip())
                    yield _sse({'type': 'log', 'message': msg.strip()})
            except asyncio.TimeoutError:
                yield _sse({'type': 'heartbeat'})

        # 推送剩余日志
        while not log_queue.empty():
            msg = log_queue.get_nowait()
            if msg.strip():
                log_lines.append(msg.strip())
                yield _sse({'type': 'log', 'message': msg.strip()})

        # 获取结果
        result: WorkerResult = result_holder.get('result')
        if result is None:
            # task 可能抛异常
            try:
                task.result()  # 触发异常
            except Exception as e:
                yield _sse({'type': 'error', 'message': str(e)})
                return

        # 保存完整日志到文件
        query_label = f"{company_info['name']} ({industry} @ {company_info['region']})"
        log_file = save_search_log("smart", query_label, log_lines, result)
        yield _sse({'type': 'log', 'message': f'💾 日志已保存: {log_file}'})

        yield _sse({'type': 'result', 'data': result.to_sse_result()})
        yield _sse({'type': 'done'})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(data: dict) -> str:
    """格式化 SSE 消息"""
    return f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "0.14", "time": datetime.now().isoformat()}


@app.get("/api/logs")
async def get_logs(n: int = Query(80, description="行数")):
    """读取服务器日志（不依赖 SSH）"""
    import subprocess
    # 读 server.log
    try:
        r1 = subprocess.run(["tail", f"-{n}", "/tmp/server.log"], capture_output=True, text=True, timeout=5)
        server_log = r1.stdout
    except Exception as e:
        server_log = f"Error: {e}"
    # 读 browser_use_debug.log
    try:
        r2 = subprocess.run(["tail", f"-{n}", "/tmp/browser_use_debug.log"], capture_output=True, text=True, timeout=5)
        debug_log = r2.stdout
    except Exception as e:
        debug_log = f"Error: {e}"
    return {"server_log": server_log, "debug_log": debug_log}


@app.get("/api/search-logs")
async def list_search_logs():
    """列出所有搜索日志文件"""
    logs = []
    for f in sorted(SEARCH_LOG_DIR.glob("search_*.log"), reverse=True):
        logs.append({
            "filename": f.name,
            "size": f.stat().st_size,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        })
    return {"logs": logs, "count": len(logs)}


@app.get("/api/search-logs/{filename}")
async def get_search_log(filename: str):
    """读取某个搜索日志的完整内容"""
    filepath = SEARCH_LOG_DIR / filename
    if not filepath.exists() or not filepath.name.startswith("search_"):
        return JSONResponse(status_code=404, content={"error": "日志不存在"})
    content = filepath.read_text(encoding="utf-8")
    return {"filename": filename, "content": content}


if __name__ == "__main__":
    import uvicorn
    print("🚀 Policy Search API v0.14 (Orchestrator 智能搜索)")
    print("   http://0.0.0.0:8000")
    print("   智能搜索: /api/policy-search/stream?industry=光通信&region=上海")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
