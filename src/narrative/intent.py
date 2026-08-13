# src/narrative/intent.py

from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, Any, Tuple, Optional
from uuid import UUID, uuid4

from src.narrative.schema import NARRATIVE_SCHEMA_VERSION
from src.narrative._utils import parse_uuid
from src.narrative.intent.dimension import IntentDimension


class IntentSource(StrEnum):
    EDITORIAL = "editorial"
    READER = "reader"
    GENRE = "genre"
    AUTHOR = "author"
    MARKET = "market"
    SYSTEM = "system"


class IntentPriority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class NarrativeIntent:
    """
    叙事意图 — 声明式目标（Phase 9.2.1 语义升级）

    Phase 9.2.1 核心变化：
    - 新增 dimension 字段（必需），实现语义化
    - desired_effect 保留为人类可读描述
    - 旧格式由 LegacyIntentLoader 负责迁移

    (See ADR-046: Intent never references implementation.)
    """
    source: IntentSource
    dimension: IntentDimension  # 必需：语义维度
    desired_effect: str         # 人类可读描述
    preserve: Tuple[str, ...] = field(default_factory=tuple)
    avoid: Tuple[str, ...] = field(default_factory=tuple)
    priority: IntentPriority = IntentPriority.MEDIUM
    rationale: str = ""
    id: UUID = field(default_factory=uuid4)

    SCHEMA_VERSION: ClassVar[str] = NARRATIVE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "source": self.source.value,
            "dimension": self.dimension.to_dict(),
            "desired_effect": self.desired_effect,
            "preserve": list(self.preserve),
            "avoid": list(self.avoid),
            "priority": self.priority.value,
            "rationale": self.rationale,
            "schema_version": self.SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NarrativeIntent":
        _ = data.get("schema_version", cls.SCHEMA_VERSION)
        return cls(
            id=parse_uuid(data.get("id"), "intent_id"),
            source=IntentSource(data["source"]),
            dimension=IntentDimension.from_dict(data["dimension"]),
            desired_effect=data.get("desired_effect", ""),
            preserve=tuple(data.get("preserve", [])),
            avoid=tuple(data.get("avoid", [])),
            priority=IntentPriority(data.get("priority", "medium")),
            rationale=data.get("rationale", ""),
        )


@dataclass(frozen=True)
class NarrativeIntentSet:
    intents: Tuple[NarrativeIntent, ...] = field(default_factory=tuple)
    SCHEMA_VERSION: ClassVar[str] = NARRATIVE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "intents": [i.to_dict() for i in self.intents],
            "schema_version": self.SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NarrativeIntentSet":
        return cls(
            intents=tuple(NarrativeIntent.from_dict(i) for i in data.get("intents", [])),
        )

    def __len__(self) -> int:
        return len(self.intents)

    def __bool__(self) -> bool:
        return bool(self.intents)

    def __iter__(self):
        return iter(self.intents)