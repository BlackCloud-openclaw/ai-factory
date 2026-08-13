# src/narrative/conflict/model.py

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Tuple, Optional, Any, Mapping
from types import MappingProxyType
from uuid import UUID, uuid4

from src.narrative.intent.dimension import IntentDirection
from src.narrative.intent.conflict import ConflictType  # 仅用于冲突模型


# ============================================================
# 现有模型（ConflictStrategy, ConflictResolution）
# ============================================================

class ConflictStrategy(StrEnum):
    PRIORITY = "priority"
    BALANCE = "balance"
    SYNTHESIS = "synthesis"
    ASK = "ask"


def parse_required_uuid(value: Optional[str], field_name: str) -> UUID:
    if not value:
        raise ValueError(f"Missing required UUID field: {field_name}")
    try:
        return UUID(value)
    except ValueError as e:
        raise ValueError(f"Invalid UUID for {field_name}: {value}") from e


def parse_uuid_optional(value: Optional[str]) -> Optional[UUID]:
    if not value:
        return None
    try:
        return UUID(value)
    except ValueError as e:
        raise ValueError(f"Invalid UUID: {value}") from e


@dataclass(frozen=True)
class ConflictResolution:
    conflict_id: UUID
    strategy: ConflictStrategy
    rationale: str = ""
    selected_intent: Optional[UUID] = None
    chosen_direction: Optional[IntentDirection] = None
    affected_intents: Tuple[UUID, ...] = field(default_factory=tuple)
    resolution_id: UUID = field(default_factory=uuid4)

    def to_dict(self) -> dict:
        return {
            "resolution_id": str(self.resolution_id),
            "conflict_id": str(self.conflict_id),
            "strategy": self.strategy.value,
            "selected_intent": str(self.selected_intent) if self.selected_intent else None,
            "chosen_direction": self.chosen_direction.value if self.chosen_direction else None,
            "rationale": self.rationale,
            "affected_intents": [str(i) for i in self.affected_intents],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConflictResolution":
        return cls(
            resolution_id=parse_required_uuid(data.get("resolution_id"), "resolution_id"),
            conflict_id=parse_required_uuid(data.get("conflict_id"), "conflict_id"),
            strategy=ConflictStrategy(data.get("strategy", "priority")),
            chosen_direction=IntentDirection(data["chosen_direction"]) if data.get("chosen_direction") else None,
            selected_intent=parse_uuid_optional(data.get("selected_intent")),
            rationale=data.get("rationale", ""),
            affected_intents=tuple(UUID(i) for i in data.get("affected_intents", []) if i),
        )

    def to_prompt(self) -> str:
        lines = [f"冲突策略: {self.strategy.value}"]
        if self.rationale:
            lines.append(f"决策理由: {self.rationale}")
        if self.chosen_direction:
            lines.append(f"采用方向: {self.chosen_direction.value}")
        return "\n".join(lines)


# ============================================================
# 🆕 新增: StrategyDecision (从 adaptive.model 移入)
# ============================================================

@dataclass(frozen=True)
class StrategyDecision:
    strategy: ConflictStrategy
    confidence: float
    reason: str
    selected_by: str
    historical_score: Optional[float] = None