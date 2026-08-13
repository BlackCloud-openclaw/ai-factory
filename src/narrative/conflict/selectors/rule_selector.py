# src/narrative/conflict/selectors/rule_selector.py

from typing import Tuple

from src.narrative.intent.model import NarrativeIntent
from src.narrative.intent.conflict import Conflict
from src.narrative.conflict import ConflictStrategy
from src.narrative.conflict.provider import StrategyDecisionProvider
from src.narrative.conflict.model import StrategyDecision


class RuleSelector(StrategyDecisionProvider):
    """
    纯规则选择器（与 Phase 9.3.2 行为一致）
    现在归属于 conflict 包，作为默认决策提供者。
    """

    def decide(
        self,
        conflicts: Tuple[Conflict, ...],
        intents: Tuple[NarrativeIntent, ...],
    ) -> StrategyDecision:
        if not conflicts:
            return StrategyDecision(
                strategy=ConflictStrategy.PRIORITY,
                confidence=1.0,
                reason="No conflicts, fallback to PRIORITY",
                selected_by="rule_fallback",
            )

        features = self._analyze_features(conflicts, intents)

        if features.get("is_absolute_conflict"):
            return StrategyDecision(
                strategy=ConflictStrategy.PRIORITY,
                confidence=1.0,
                reason="Absolute conflict detected → PRIORITY",
                selected_by="rule",
            )

        if features.get("has_priority_diff"):
            return StrategyDecision(
                strategy=ConflictStrategy.PRIORITY,
                confidence=1.0,
                reason="Priority difference detected → PRIORITY",
                selected_by="rule",
            )

        if features.get("synthesis_hint"):
            return StrategyDecision(
                strategy=ConflictStrategy.SYNTHESIS,
                confidence=1.0,
                reason="Synthesis hint detected → SYNTHESIS",
                selected_by="rule",
            )

        return StrategyDecision(
            strategy=ConflictStrategy.BALANCE,
            confidence=1.0,
            reason="Default → BALANCE",
            selected_by="rule",
        )

    def _analyze_features(self, conflicts, intents) -> dict:
        features = {
            "has_priority_diff": False,
            "is_absolute_conflict": False,
            "synthesis_hint": False,
        }

        for conflict in conflicts:
            conflict_ids = [i.id for i in conflict.intents]
            conflict_intents = [i for i in intents if i.id in conflict_ids]

            if len(conflict_intents) >= 2:
                priorities = [i.priority for i in conflict_intents]
                if len(set(priorities)) > 1:
                    features["has_priority_diff"] = True

            if conflict.metadata.get("resolution_hint") == "absolute":
                features["is_absolute_conflict"] = True

            if conflict.metadata.get("synthesis_hint") is True:
                features["synthesis_hint"] = True

        return features