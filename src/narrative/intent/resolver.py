# src/narrative/intent/resolver.py

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Tuple, Mapping, Any, Optional

from src.narrative.intent.model import NarrativeIntent, NarrativeIntentSet, IntentPriority
from src.narrative.intent.conflict import Conflict, detect_direction_conflicts
from src.narrative.conflict import create_resolver, ConflictResolver, ConflictResolution


class ResolutionStrategy(StrEnum):
    PRIORITY_BASED = "priority_based"


@dataclass(frozen=True)
class ResolutionPlan:
    primary_intents: Tuple[NarrativeIntent, ...] = field(default_factory=tuple)
    conflicts: Tuple[Conflict, ...] = field(default_factory=tuple)
    resolutions: Tuple[ConflictResolution, ...] = field(default_factory=tuple)
    strategy: ResolutionStrategy = ResolutionStrategy.PRIORITY_BASED
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict:
        return {
            "primary_intents": [i.to_dict() for i in self.primary_intents],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "resolutions": [r.to_dict() for r in self.resolutions],
            "strategy": self.strategy.value,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ResolutionPlan":
        from src.narrative.intent.model import NarrativeIntent
        # 注意：Conflict 的 from_dict 只重建身份，intents 列表为空
        intents = tuple(NarrativeIntent.from_dict(i) for i in data.get("primary_intents", []))
        conflicts = tuple(Conflict.from_dict(c) for c in data.get("conflicts", []))
        resolutions = tuple(ConflictResolution.from_dict(r) for r in data.get("resolutions", []))
        strategy = ResolutionStrategy(data.get("strategy", "priority_based"))
        metadata = data.get("metadata", {})
        return cls(
            primary_intents=intents,
            conflicts=conflicts,
            resolutions=resolutions,
            strategy=strategy,
            metadata=metadata,
        )

    @property
    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0

    @property
    def conflict_count(self) -> int:
        return len(self.conflicts)


class ConflictResolutionError(RuntimeError):
    pass


class IntentResolver:
    def __init__(self, conflict_resolver: Optional[ConflictResolver] = None):
        self._conflict_resolver = conflict_resolver or create_resolver("priority")

    def resolve(self, intents: NarrativeIntentSet) -> ResolutionPlan:
        if not intents:
            return ResolutionPlan(
                primary_intents=(),
                conflicts=(),
                resolutions=(),
                strategy=ResolutionStrategy.PRIORITY_BASED,
                metadata={"reason": "no_intents"},
            )

        sorted_intents = tuple(
            sorted(intents.intents, key=self._priority_weight, reverse=True)
        )

        conflicts = tuple(detect_direction_conflicts(sorted_intents))

        resolutions = self._conflict_resolver.resolve(conflicts, sorted_intents)

        if len(resolutions) != len(conflicts):
            raise ConflictResolutionError(
                f"ConflictResolver violated completeness invariant: "
                f"{len(resolutions)} resolutions for {len(conflicts)} conflicts"
            )

        metadata = {
            "total_intents": len(sorted_intents),
            "conflict_count": len(conflicts),
            "strategy": ResolutionStrategy.PRIORITY_BASED.value,
        }

        return ResolutionPlan(
            primary_intents=sorted_intents,
            conflicts=conflicts,
            resolutions=resolutions,
            strategy=ResolutionStrategy.PRIORITY_BASED,
            metadata=metadata,
        )

    @staticmethod
    def _priority_weight(intent: NarrativeIntent) -> int:
        weights = {
            IntentPriority.HIGH: 3,
            IntentPriority.MEDIUM: 2,
            IntentPriority.LOW: 1,
        }
        return weights.get(intent.priority, 0)


def resolve_intents(intents: NarrativeIntentSet) -> ResolutionPlan:
    return IntentResolver().resolve(intents)