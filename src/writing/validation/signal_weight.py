# src/writing/validation/signal_weight.py
"""
Signal Weight Policy

Phase 13.2.3B: B-2
"""

from src.writing.planning_contract import SignalSource


class SignalWeightPolicy:
    """信号权重策略 - 根据来源返回验证置信度权重。"""

    DEFAULT_WEIGHTS = {
        SignalSource.LLM: 1.0,
        SignalSource.SYSTEM: 0.8,
        SignalSource.INFERRED: 0.6,
        SignalSource.NORMALIZED: 0.5,
        SignalSource.UNKNOWN: 0.3,
    }

    def __init__(self, weights: dict = None):
        self._weights = weights or self.DEFAULT_WEIGHTS.copy()

    def weight(self, source: SignalSource) -> float:
        return self._weights.get(source, 0.3)

    def weighted_score(self, base_score: float, source: SignalSource) -> float:
        return base_score * self.weight(source)

    def update_weight(self, source: SignalSource, new_weight: float) -> None:
        self._weights[source] = max(0.0, min(1.0, new_weight))