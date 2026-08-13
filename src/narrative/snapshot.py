# src/narrative/snapshot.py

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping
from uuid import UUID, uuid4

from src.narrative.schema import NARRATIVE_SCHEMA_VERSION
from src.narrative._utils import parse_uuid


@dataclass(frozen=True)
class StorySnapshot:
    snapshot_id: UUID = field(default_factory=uuid4)
    projection: Any = field(default_factory=dict)

    SCHEMA_VERSION: ClassVar[str] = NARRATIVE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": str(self.snapshot_id),
            "projection": self.projection,
            "schema_version": self.SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StorySnapshot":
        _ = data.get("schema_version", cls.SCHEMA_VERSION)
        return cls(
            snapshot_id=parse_uuid(data.get("snapshot_id"), "snapshot_id"),
            projection=data.get("projection", {}),
        )