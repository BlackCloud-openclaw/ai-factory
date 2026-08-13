# src/narrative/constraint.py

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping
from uuid import UUID, uuid4

from src.narrative.schema import NARRATIVE_SCHEMA_VERSION
from src.narrative._utils import parse_uuid


@dataclass(frozen=True)
class NarrativeConstraint:
    constraint_id: UUID = field(default_factory=uuid4)
    payload: Any = field(default_factory=dict)

    SCHEMA_VERSION: ClassVar[str] = NARRATIVE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": str(self.constraint_id),
            "payload": self.payload,
            "schema_version": self.SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NarrativeConstraint":
        _ = data.get("schema_version", cls.SCHEMA_VERSION)
        return cls(
            constraint_id=parse_uuid(data.get("constraint_id"), "constraint_id"),
            payload=data.get("payload", {}),
        )