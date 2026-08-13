# src/narrative/conflict/composite.py

from typing import Tuple, Optional, Dict

from src.narrative.intent.model import NarrativeIntent
from src.narrative.intent.conflict import Conflict
from src.narrative.conflict.model import ConflictResolution, StrategyDecision
from src.narrative.conflict.protocol import ConflictResolver
from src.narrative.conflict.strategies import (
    PriorityResolver,
    BalanceResolver,
    SynthesisResolver,
)
from src.narrative.conflict import ConflictStrategy
from src.narrative.conflict.provider import StrategyDecisionProvider
from src.narrative.conflict.selectors import RuleSelector  # ✅ 正确导入


class CompositeResolver(ConflictResolver):
    def __init__(
        self,
        provider: Optional[StrategyDecisionProvider] = None,
    ):
        self._provider = provider or RuleSelector()  # 默认使用 RuleSelector
        self._last_decision: Optional[StrategyDecision] = None
        self._last_resolver_name: Optional[str] = None

        self._resolvers: Dict[ConflictStrategy, ConflictResolver] = {
            ConflictStrategy.PRIORITY: PriorityResolver(),
            ConflictStrategy.BALANCE: BalanceResolver(),
            ConflictStrategy.SYNTHESIS: SynthesisResolver(),
        }

    def resolve(
        self,
        conflicts: Tuple[Conflict, ...],
        intents: Tuple[NarrativeIntent, ...],
    ) -> Tuple[ConflictResolution, ...]:
        if not conflicts:
            return ()

        decision = self._provider.decide(conflicts, intents)
        self._last_decision = decision

        resolver = self._resolvers.get(decision.strategy)
        if resolver is None:
            resolver = self._resolvers[ConflictStrategy.PRIORITY]

        self._last_resolver_name = type(resolver).__name__
        return resolver.resolve(conflicts, intents)

    def get_last_decision(self) -> Optional[StrategyDecision]:
        return self._last_decision

    def get_last_resolver_name(self) -> Optional[str]:
        return self._last_resolver_name