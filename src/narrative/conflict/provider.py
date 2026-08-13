# src/narrative/conflict/provider.py

from typing import Protocol, Tuple, runtime_checkable

from src.narrative.intent.model import NarrativeIntent
from src.narrative.intent.conflict import Conflict
from src.narrative.conflict.model import StrategyDecision  # ✅ 从同包 model 导入


@runtime_checkable
class StrategyDecisionProvider(Protocol):
    def decide(
        self,
        conflicts: Tuple[Conflict, ...],
        intents: Tuple[NarrativeIntent, ...],
    ) -> StrategyDecision:
        ...