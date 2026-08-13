# src/narrative/adaptive/provider.py

from typing import Protocol, Tuple, runtime_checkable
from src.narrative.intent.model import NarrativeIntent
from src.narrative.intent.conflict import Conflict
from src.narrative.adaptive.model import StrategyDecision


@runtime_checkable
class StrategyDecisionProvider(Protocol):
    def decide(
        self,
        conflicts: Tuple[Conflict, ...],
        intents: Tuple[NarrativeIntent, ...],
    ) -> StrategyDecision:
        ...