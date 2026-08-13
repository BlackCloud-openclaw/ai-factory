# src/narrative/compilers/interfaces.py

"""
IntentCompiler 接口定义

依赖方向：
src.runtime.observation.interfaces
    ↑
src.narrative.compilers.interfaces
    ↑
src.narrative.compilers.intent_compiler
    ↑
src.narrative.compilers.analyzers
"""

from typing import List, Protocol, runtime_checkable

from src.narrative.intent import NarrativeIntent
from src.runtime.observation.interfaces import ObservationProtocol


@runtime_checkable
class IntentAnalyzer(Protocol):
    """
    意图分析器协议

    每个 Analyzer 负责一个维度的 Intent 生成。
    由 Composition Root 注册。
    """

    def analyze(
        self,
        observation: ObservationProtocol,
        context: dict,
    ) -> List[NarrativeIntent]:
        """
        分析观察结果，生成 Intent 列表

        Args:
            observation: Phase 8 Observation 协议
            context: 叙事上下文（章节、场景等元信息）

        Returns:
            List[NarrativeIntent]: 生成的意图列表（可能为空）
        """
        ...