"""
统一数据模型 & Worker 基类
=========================
所有 Worker 共用的数据模型和接口定义。

PolicyItem   — 单条政策
WorkerResult — Worker 统一输出
BaseWorker   — Worker 抽象基类（search → WorkerResult）

设计原则：
    1. 前端只认一种格式 — WorkerResult.policies[]
    2. Server 零转换 — worker.search(query) 直接用
    3. Worker 随意切换 — 都实现 BaseWorker.search()
"""

from __future__ import annotations

import json
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 统一数据模型
# ─────────────────────────────────────────────

@dataclass
class PolicyItem:
    """单条政策信息 — 前端直接渲染的最小单元"""
    title: str = ""              # 政策标题
    url: str = ""                # 原文链接
    source: str = ""             # 发布机构
    date: str = ""               # 发布日期
    summary: str = ""            # 摘要
    support: str = ""            # 扶持内容（金额/比例等）
    pdf_url: str = ""            # PDF 下载链接
    industry: str = ""           # 适用行业
    full_text: str = ""          # 全文（深度模式才有）
    layer: str = ""              # 业务层：基础层/发展层/人才层/荣誉层
    relevance: int = 0           # 综合评分（0-100，由5维度加权计算）
    score_amount: int = 0        # 💰 金额价值评分（0-100）
    score_exclusivity: int = 0   # 🎯 独占性评分（0-100）
    score_feasibility: int = 0   # ✅ 可得性评分（0-100）
    score_urgency: int = 0       # ⏰ 紧迫性评分（0-100）
    score_sustainability: int = 0  # 🔄 持续性评分（0-100）
    score_reason: str = ""       # 评分理由
    validity: str = ""           # 有效期（如 "2025-12-31" 或 "长期有效"）
    application_deadline: str = ""  # 申报截止日期（如 "2025-03-28"）
    amount: str = ""             # 金额范围（如 "最高20万" "最高1000万" "税率减半"）
    amount_level: str = ""       # 金额等级：S/A/B/C/D


@dataclass
class WorkerResult:
    """
    Worker 统一输出 — 所有 Worker 的 search() 都返回这个

    前端需要的字段:
        policies     → 渲染政策列表
        sources      → 显示引用链接
        worker       → 显示是哪个 Worker 产出的
        duration     → 显示耗时
        error        → 显示错误

    内部字段:
        query        → 回溯用
        raw_answer   → 调试用（LLM 原始回答）
        token_usage  → 成本追踪
    """
    query: str = ""                                    # 原始查询
    policies: list[PolicyItem] = field(default_factory=list)  # 政策列表
    sources: list[str] = field(default_factory=list)   # 所有引用 URL
    worker: str = ""                                   # 产出者: "web_search" | "browser_use"
    duration: float = 0.0                              # 耗时(秒)
    token_usage: dict = field(default_factory=dict)    # Token 消耗
    error: Optional[str] = None                        # 错误信息
    raw_answer: str = ""                               # LLM 原始回答（调试用）

    @property
    def success(self) -> bool:
        return self.error is None and len(self.policies) > 0

    @property
    def policy_count(self) -> int:
        return len(self.policies)

    def to_dict(self) -> dict:
        """转为 dict（前端/SSE 直接用）"""
        d = asdict(self)
        d["success"] = self.success
        d["policy_count"] = self.policy_count
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_sse_result(self) -> dict:
        """
        转为 SSE 兼容格式 — 保持和原 server.py 的 {type:'result', data:{...}} 兼容

        前端 JS 解析路径:
            data.data.policies → 政策列表
            data.data.duration → 耗时
            data.data.worker   → 哪个 Worker
        """
        return self.to_dict()


# ─────────────────────────────────────────────
# Worker 基类
# ─────────────────────────────────────────────

class BaseWorker(ABC):
    """
    Worker 抽象基类

    所有 Worker 必须实现:
        name       — Worker 名称标识
        search()   — 搜索，返回 WorkerResult
    """

    name: str = "base"

    @abstractmethod
    def search(self, query: str, **kwargs) -> WorkerResult:
        """
        执行搜索

        Args:
            query: 搜索查询（自然语言）
            **kwargs: Worker 特有参数

        Returns:
            WorkerResult
        """
        ...

    def timed_search(self, query: str, **kwargs) -> WorkerResult:
        """带计时的搜索（自动填充 duration 和 worker 字段）"""
        start = time.time()
        result = self.search(query, **kwargs)
        result.duration = round(time.time() - start, 1)
        result.worker = self.name
        return result
