# src/writing/snapshot/runtime/remote/gc/retention_checker.py
"""
B4.10: ChunkRetentionChecker — 判断 Chunk 是否应被保留
"""

from enum import Enum, auto
from typing import Protocol

from ...chunk_ref import ChunkRef
from .retention import RetentionPolicy
from .reachability import ReachabilityAnalyzer
from .capability import ChunkEnumerator
from .models import ReachabilityGraph


class RetentionDecision(Enum):
    """保留决策结果。"""
    RETAIN = auto()      # 应保留（因可达性或策略）
    DELETE = auto()      # 应删除（不被任何保留策略覆盖）


class ChunkRetentionChecker(Protocol):
    """
    检查 Chunk 的保留决策（纯能力抽象）。

    语义：
        - 返回 RETAIN：该 Chunk 应保留（不应删除）。
        - 返回 DELETE：该 Chunk 应删除（不被任何保留策略覆盖）。
    """

    def decide(self, ref: ChunkRef) -> RetentionDecision:
        """
        返回 Chunk 的保留决策。

        Args:
            ref: Chunk 引用。

        Returns:
            RetentionDecision 枚举值。
        """
        ...


class RetentionPolicyBasedChecker:
    """
    基于 RetentionPolicy + ReachabilityAnalyzer 的检查器。

    决策逻辑：
        1. 如果 chunk 在可达图中，返回 RETAIN
        2. 否则返回 DELETE
    """

    def __init__(
        self,
        chunk_enumerator: ChunkEnumerator,
        retention_policy: RetentionPolicy,
        version_store,
    ):
        self._chunk_enumerator = chunk_enumerator
        self._retention_policy = retention_policy
        self._version_store = version_store
        self._graph: ReachabilityGraph | None = None

    def _ensure_graph(self) -> None:
        if self._graph is None:
            analyzer = ReachabilityAnalyzer(
                self._version_store,
                self._chunk_enumerator,
                self._retention_policy,
            )
            self._graph = analyzer.analyze()

    def decide(self, ref: ChunkRef) -> RetentionDecision:
        self._ensure_graph()
        if ref in self._graph.reachable_chunks:
            return RetentionDecision.RETAIN
        return RetentionDecision.DELETE