"""
Web Search Worker
=================
使用 Azure OpenAI Responses API + web_search_preview 工具的搜索 Worker。
不需要创建额外的 Bing 资源，直接调用 Responses API。

实现 BaseWorker 接口，search() 返回统一的 WorkerResult。

官方文档:
    https://learn.microsoft.com/azure/ai-foundry/openai/how-to/web-search

环境变量:
    AZURE_AI_PROJECT_ENDPOINT        - Azure AI Foundry 项目端点
    AZURE_AI_API_KEY                 - API Key
    AZURE_AI_MODEL_DEPLOYMENT_NAME   - 模型部署名称 (如 gpt-4o)

使用方式:
    1. 命令行搜索:
       python web_search_worker.py "浦东新区光通信产业扶持政策"

    2. 作为模块导入:
       from web_search_worker import WebSearchWorker
       worker = WebSearchWorker()
       result = worker.search("浦东新区光通信产业扶持政策")  # → WorkerResult

    3. 作为 FastAPI 服务:
       python web_search_worker.py --serve --port 8001
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # 加载 .env
load_dotenv(".env.web_search")  # 加载 .env.web_search (覆盖)

from models import BaseWorker, WorkerResult, PolicyItem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 默认 Instructions（要求输出链接、PDF）
# ─────────────────────────────────────────────

DEFAULT_POLICY_INSTRUCTIONS = """你是一个专业的政策研究助手。请根据搜索结果，尽可能完整地提取和呈现政策信息。

要求：
1. 提供每条政策的完整标题、发文字号、发布日期、发布机构
2. 列出政策的具体条款和措施（原文摘录，不要概括）
3. 包含具体的数字、比例、金额上限等关键数据
4. 每条政策都必须给出官网原文链接（URL）
5. 如果有PDF下载链接，也要列出
6. 标注每条政策适用的行业范围
7. 搜索尽可能多的相关来源，广泛覆盖政府官网、政策库等渠道

输出格式要求（严格JSON，不要输出其他文字）：
{
  "policies": [
    {
      "title": "政策完整标题",
      "source": "发布机构",
      "url": "官网原文链接",
      "pdf_url": "PDF下载链接（没有则留空）",
      "date": "发布日期",
      "summary": "政策摘要（包含具体数字和比例）",
      "support": "关键扶持内容（资金额度、补贴比例等）",
      "industry": "适用行业"
    }
  ]
}"""


# ─────────────────────────────────────────────
# Web Search Worker
# ─────────────────────────────────────────────

class WebSearchWorker(BaseWorker):
    """
    Azure OpenAI Responses API + web_search_preview Worker

    直接使用 Responses API 的内置 web_search_preview 工具，
    无需创建 Agent，无需 azure-ai-projects 包。

    实现 BaseWorker 接口:
        worker = WebSearchWorker()
        result = worker.search("查询内容")  # → WorkerResult
    """

    name = "web_search"

    def __init__(
        self,
        endpoint: str = None,
        api_key: str = None,
        model_deployment: str = None,
        api_version: str = "2025-04-01-preview",
        instructions: str = None,
        search_context_size: str = "high",
    ):
        self.api_key = api_key or os.environ.get("AZURE_AI_API_KEY")
        self.model_deployment = model_deployment or os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o")
        self.api_version = api_version
        self.search_context_size = search_context_size  # "low" | "medium" | "high"
        self.instructions = instructions or DEFAULT_POLICY_INSTRUCTIONS

        # 从 project endpoint 提取 OpenAI endpoint
        project_endpoint = endpoint or os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "")
        self.endpoint = self._resolve_openai_endpoint(project_endpoint)

        if not self.endpoint:
            raise ValueError(
                "需要设置 AZURE_AI_PROJECT_ENDPOINT 环境变量，"
                "或在初始化时传入 endpoint 参数。\n"
                "获取方式: Azure AI Foundry Portal → Project → Settings → Overview"
            )
        if not self.api_key:
            raise ValueError(
                "需要设置 AZURE_AI_API_KEY 环境变量，"
                "或在初始化时传入 api_key 参数。"
            )

        self._client = None

    @staticmethod
    def _resolve_openai_endpoint(project_endpoint: str) -> str:
        """
        从 Foundry Project Endpoint 提取 Azure OpenAI 兼容 Endpoint。
        例: https://xxx-resource.services.ai.azure.com/api/projects/xxx
          → https://xxx-resource.services.ai.azure.com
        """
        if not project_endpoint:
            return ""
        # 去掉 /api/projects/xxx 部分，保留基础 URL
        from urllib.parse import urlparse
        parsed = urlparse(project_endpoint)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _ensure_client(self):
        """延迟初始化 OpenAI 客户端"""
        if self._client is not None:
            return

        from openai import AzureOpenAI

        logger.info(f"初始化 AzureOpenAI 客户端 (endpoint: {self.endpoint})...")
        self._client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
        )
        logger.info("客户端初始化完成")

    def search(self, query: str, **kwargs) -> WorkerResult:
        """
        执行搜索，返回统一的 WorkerResult（实现 BaseWorker 接口）

        Args:
            query: 搜索查询

        Returns:
            WorkerResult（包含 policies, sources 等）
        """
        start = time.time()
        self._ensure_client()

        logger.info(f"[web_search] 搜索: {query}")

        try:
            response = self._client.responses.create(
                model=self.model_deployment,
                instructions=self.instructions,
                tools=[{"type": "web_search_preview", "search_context_size": self.search_context_size}],
                input=query,
            )

            # 提取回答文本
            answer = response.output_text or ""

            # 提取引用 URL（去重）
            sources = []
            seen_urls = set()
            for item in response.output:
                if hasattr(item, "content"):
                    for content in item.content:
                        if hasattr(content, "annotations"):
                            for ann in content.annotations:
                                if hasattr(ann, "type") and ann.type == "url_citation":
                                    if ann.url not in seen_urls:
                                        seen_urls.add(ann.url)
                                        sources.append(ann.url)

            # 提取用量信息
            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "input_tokens": getattr(response.usage, "input_tokens", 0),
                    "output_tokens": getattr(response.usage, "output_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                }

            # 解析 LLM 回答 → PolicyItem 列表
            policies = self._parse_policies(answer, sources)

            elapsed = round(time.time() - start, 1)
            logger.info(f"[web_search] 完成, 政策数: {len(policies)}, 引用数: {len(sources)}, 耗时: {elapsed}s")

            return WorkerResult(
                query=query,
                policies=policies,
                sources=sources,
                worker=self.name,
                duration=elapsed,
                token_usage=usage,
                raw_answer=answer,
            )

        except Exception as e:
            elapsed = round(time.time() - start, 1)
            logger.error(f"[web_search] 搜索失败: {e}")
            return WorkerResult(
                query=query,
                worker=self.name,
                duration=elapsed,
                error=str(e),
            )

    @staticmethod
    def _parse_policies(answer: str, sources: list[str]) -> list[PolicyItem]:
        """
        从 LLM 回答中解析 PolicyItem 列表。
        优先尝试 JSON 解析，失败则用引用 URL 构建基础列表。
        """
        # 尝试从回答中提取 JSON
        json_match = re.search(r'\{[\s\S]*"policies"[\s\S]*\}', answer)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                items = []
                for p in parsed.get("policies", []):
                    items.append(PolicyItem(
                        title=p.get("title") or p.get("policy_title", ""),
                        url=p.get("url", ""),
                        source=p.get("source", ""),
                        date=p.get("date") or p.get("publish_date", ""),
                        summary=p.get("summary", ""),
                        support=p.get("support") or p.get("key_support", ""),
                        pdf_url=p.get("pdf_url", ""),
                        industry=p.get("industry") or p.get("applicable_industry", ""),
                    ))
                if items:
                    return items
            except json.JSONDecodeError:
                pass

        # JSON 解析失败：用引用 URL 构建基础列表
        if sources:
            return [
                PolicyItem(title=f"搜索结果 {i+1}", url=url)
                for i, url in enumerate(sources)
            ]

        return []

    def search_stream(self, query: str):
        """
        流式搜索，逐步 yield 文本片段

        Args:
            query: 搜索查询

        Yields:
            dict: {"type": "delta"|"citation"|"done", "content": ...}
        """
        self._ensure_client()

        logger.info(f"流式搜索: {query}")

        try:
            stream_response = self._client.responses.create(
                model=self.model_deployment,
                instructions=self.instructions,
                tools=[{"type": "web_search_preview", "search_context_size": self.search_context_size}],
                input=query,
                stream=True,
            )

            for event in stream_response:
                if event.type == "response.output_text.delta":
                    yield {"type": "delta", "content": event.delta}

                elif event.type == "response.output_item.done":
                    if event.item.type == "message":
                        text_content = event.item.content[-1]
                        if hasattr(text_content, "annotations"):
                            for ann in text_content.annotations:
                                if ann.type == "url_citation":
                                    yield {
                                        "type": "citation",
                                        "content": {
                                            "url": ann.url,
                                            "title": getattr(ann, "title", ""),
                                        },
                                    }

                elif event.type == "response.completed":
                    yield {"type": "done", "content": ""}

        except Exception as e:
            logger.error(f"流式搜索失败: {e}")
            yield {"type": "error", "content": str(e)}

    def close(self):
        """关闭客户端"""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("客户端已关闭")


# ─────────────────────────────────────────────
# FastAPI 服务模式
# ─────────────────────────────────────────────

def create_app() -> "FastAPI":
    """创建 FastAPI 应用"""
    from fastapi import FastAPI, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, JSONResponse

    app = FastAPI(title="Web Search Worker API", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    worker = WebSearchWorker()

    @app.get("/search")
    async def search(q: str = Query(..., description="搜索查询")):
        """同步搜索接口，返回完整结果"""
        result = worker.search(q)
        return JSONResponse(content=result.to_dict())

    @app.get("/search/stream")
    async def search_stream(q: str = Query(..., description="搜索查询")):
        """流式搜索接口，SSE 格式"""
        def event_generator():
            for chunk in worker.search_stream(q):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


# ─────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Web Search Worker")
    parser.add_argument("query", nargs="?", help="搜索查询内容")
    parser.add_argument("--serve", action="store_true", help="启动 FastAPI 服务")
    parser.add_argument("--port", type=int, default=8001, help="服务端口 (默认 8001)")
    parser.add_argument("--stream", action="store_true", help="使用流式输出")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")
    parser.add_argument("--context", choices=["low", "medium", "high"], default="medium", help="搜索上下文量 (默认 medium, high=更多链接)")
    args = parser.parse_args()

    # 服务模式
    if args.serve:
        import uvicorn
        app = create_app()
        logger.info(f"启动 Web Search Worker API 服务，端口: {args.port}")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
        return

    # 搜索模式
    if not args.query:
        parser.print_help()
        return

    worker = WebSearchWorker(search_context_size=args.context)

    try:
        if args.stream:
            # 流式输出
            for chunk in worker.search_stream(args.query):
                if chunk["type"] == "delta":
                    print(chunk["content"], end="", flush=True)
                elif chunk["type"] == "citation":
                    print(f"\n📎 {chunk['content']['url']}")
                elif chunk["type"] == "done":
                    print("\n\n✅ 完成")
        else:
            # 统一输出
            result = worker.search(args.query)

            if args.json:
                print(result.to_json())
            else:
                print(f"\n{'='*60}")
                print(f"🔍 查询: {result.query}")
                print(f"{'='*60}")

                if result.error:
                    print(f"\n❌ 错误: {result.error}")
                else:
                    print(f"\n📝 找到 {result.policy_count} 条政策 (Worker: {result.worker}, 耗时: {result.duration}s)")
                    for i, p in enumerate(result.policies, 1):
                        print(f"\n  {i}. {p.title}")
                        if p.source: print(f"     来源: {p.source}")
                        if p.date:   print(f"     日期: {p.date}")
                        if p.summary: print(f"     摘要: {p.summary[:100]}...")
                        if p.support: print(f"     💰 {p.support}")
                        if p.url:     print(f"     📄 {p.url}")
                        if p.pdf_url: print(f"     📥 {p.pdf_url}")

                    if result.sources:
                        print(f"\n📎 引用来源 ({len(result.sources)} 个):")
                        for i, url in enumerate(result.sources, 1):
                            print(f"   {i}. {url}")

                    if result.token_usage:
                        print(f"\n📊 Token: {result.token_usage}")
    finally:
        worker.close()


if __name__ == "__main__":
    asyncio.run(main())
