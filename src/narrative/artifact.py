# src/narrative/artifact.py

from dataclasses import dataclass, field
from typing import ClassVar, Mapping, Any
from uuid import UUID, uuid4

from src.narrative.schema import NARRATIVE_SCHEMA_VERSION
from src.narrative._utils import parse_uuid


@dataclass(frozen=True)
class NarrativeArtifact:
    text: str
    artifact_id: UUID = field(default_factory=uuid4)

    SCHEMA_VERSION: ClassVar[str] = NARRATIVE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, str]:
        return {
            "text": self.text,
            "artifact_id": str(self.artifact_id),
            "schema_version": self.SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NarrativeArtifact":
        _ = data.get("schema_version", cls.SCHEMA_VERSION)
        return cls(
            text=data["text"],
            artifact_id=parse_uuid(data.get("artifact_id"), "artifact_id"),
        )