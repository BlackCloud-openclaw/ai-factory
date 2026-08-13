# src/narrative/conflict/selector.py

from typing import Tuple
from src.narrative.intent.model import NarrativeIntent, IntentPriority
from src.narrative.intent.conflict import Conflict
from src.narrative.conflict.protocol import ConflictResolver
from src.narrative.conflict.strategies import (
    PriorityResolver,
    BalanceResolver,
    SynthesisResolver,
)


class StrategySelector:
    """根据冲突特征选择解析策略"""

    def select(
        self,
        conflicts: Tuple[Conflict, ...],
        intents: Tuple[NarrativeIntent, ...],
    ) -> ConflictResolver:
        if not conflicts:
            return PriorityResolver()

        features = self._analyze_features(conflicts, intents)

        # 绝对冲突 → Priority
        if features.get("is_absolute_conflict"):
            return PriorityResolver()

        # 有优先级差异 → Priority
        if features.get("has_priority_diff"):
            return PriorityResolver()

        # 有合成提示 → Synthesis
        if features.get("synthesis_hint"):
            return SynthesisResolver()

        # 默认 → Balance
        return BalanceResolver()

    def _analyze_features(
        self,
        conflicts: Tuple[Conflict, ...],
        intents: Tuple[NarrativeIntent, ...],
    ) -> dict:
        features = {
            "has_priority_diff": False,
            "single_dimension": True,
            "is_absolute_conflict": False,
            "synthesis_hint": False,
        }

        # 检查优先级差异
        for conflict in conflicts:
            conflict_ids = [i.id for i in conflict.intents]
            conflict_intents = [i for i in intents if i.id in conflict_ids]
            if len(conflict_intents) >= 2:
                priorities = [i.priority for i in conflict_intents]
                if len(set(priorities)) > 1:
                    features["has_priority_diff"] = True

        # 检查维度
        dimensions = set()
        for conflict in conflicts:
            dim = conflict.metadata.get("dimension")
            if dim:
                dimensions.add(dim)
        features["single_dimension"] = len(dimensions) <= 1

        # 检查绝对冲突提示
        for conflict in conflicts:
            if conflict.metadata.get("resolution_hint") == "absolute":
                features["is_absolute_conflict"] = True
                break

        # 检查合成提示
        for conflict in conflicts:
            if conflict.metadata.get("synthesis_hint") is True:
                features["synthesis_hint"] = True
                break

        return features