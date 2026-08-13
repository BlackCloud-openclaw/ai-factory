# src/narrative/compilers/intent_compiler.py

"""
Intent Compiler — 将 Phase 8 Observation 编译为 NarrativeIntentSet

依赖方向：
interfaces.py → intent_compiler.py

设计原则：
- 由 Composition Root 注入 Analyzer 列表
- 不硬编码 Analyzer
- 不重新定义 Observation 协议
"""

import logging
from typing import List, Optional

from src.narrative.intent import NarrativeIntentSet
from src.narrative.compilers.interfaces import IntentAnalyzer
from src.runtime.observation.interfaces import ObservationProtocol

logger = logging.getLogger(__name__)


class IntentCompiler:
    """
    意图编译器 — 将 Observation 编译为 NarrativeIntentSet

    流程：
    1. 遍历所有注册的 Analyzer
    2. 每个 Analyzer 分析 Observation，生成 Intent
    3. 汇总所有 Intent 为 IntentSet
    """

    def __init__(self, analyzers: Optional[List[IntentAnalyzer]] = None):
        """
        Args:
            analyzers: 分析器列表（由 Composition Root 注入）
        """
        self._analyzers: List[IntentAnalyzer] = analyzers or []

    def register_analyzer(self, analyzer: IntentAnalyzer) -> None:
        """注册一个分析器（支持动态注册）"""
        self._analyzers.append(analyzer)

    def register_analyzers(self, analyzers: List[IntentAnalyzer]) -> None:
        """注册多个分析器"""
        self._analyzers.extend(analyzers)

    def compile(
        self,
        observation: ObservationProtocol,
        context: dict,
    ) -> NarrativeIntentSet:
        """
        编译 Observation 为 NarrativeIntentSet

        Args:
            observation: Phase 8 Observation（实现 ObservationProtocol）
            context: 叙事上下文（包含章节信息等）

        Returns:
            NarrativeIntentSet: 意图集合
        """
        all_intents = []

        for analyzer in self._analyzers:
            try:
                intents = analyzer.analyze(observation, context)
                all_intents.extend(intents)
            except Exception as e:
                logger.warning(
                    f"Analyzer {analyzer.__class__.__name__} failed: {e}"
                )

        return NarrativeIntentSet(intents=tuple(all_intents))