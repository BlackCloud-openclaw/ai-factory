# src/narrative/intent/conflict.py

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Tuple, Mapping, Any
from types import MappingProxyType
from uuid import UUID, uuid4

from src.narrative.intent.model import NarrativeIntent


class ConflictType(StrEnum):
    DIRECTION_MISMATCH = "direction_mismatch"


@dataclass(frozen=True)
class Conflict:
    type: ConflictType
    intents: Tuple[NarrativeIntent, ...]
    description: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self):
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "type": self.type.value,
            "intent_ids": [str(i.id) for i in self.intents],
            "description": self.description,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Conflict":
        """
        从字典重建 Conflict 身份。
        注意：intents 字段被置空，因为意图引用需要由调用者解析，
        避免从 Conflict 内部重建完整的 NarrativeIntent 形成循环依赖。
        """
        return cls(
            type=ConflictType(data["type"]),
            intents=(),  # 由调用者填充
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
            id=UUID(data["id"]) if data.get("id") else uuid4(),
        )


def detect_direction_conflicts(intents: Tuple[NarrativeIntent, ...]) -> list[Conflict]:
    conflicts = []
    for i, a in enumerate(intents):
        for b in intents[i + 1:]:
            if a.dimension.is_opposite(b.dimension):
                conflicts.append(
                    Conflict(
                        type=ConflictType.DIRECTION_MISMATCH,
                        intents=(a, b),
                        description=(
                            f"维度 '{a.dimension.id}' 方向冲突: "
                            f"{a.dimension.direction.value} vs {b.dimension.direction.value}"
                        ),
                        metadata={
                            "dimension": a.dimension.id,
                            "direction_a": a.dimension.direction.value,
                            "direction_b": b.dimension.direction.value,
                        },
                    )
                )
    return conflicts