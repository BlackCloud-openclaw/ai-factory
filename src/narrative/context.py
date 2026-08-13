# src/narrative/context.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Mapping, Tuple, Optional, TYPE_CHECKING
from uuid import UUID, uuid4

from src.narrative.schema import NARRATIVE_SCHEMA_VERSION
from src.narrative.snapshot import StorySnapshot
from src.narrative._utils import parse_uuid

if TYPE_CHECKING:
    from src.narrative.intent.conflict import Conflict
    from src.narrative.conflict.model import ConflictResolution


class ArcStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"


@dataclass(frozen=True)
class ChapterMetadata:
    volume: int
    chapter: int
    scene_index: int
    total_scenes: int

    SCHEMA_VERSION: ClassVar[str] = NARRATIVE_SCHEMA_VERSION


@dataclass(frozen=True)
class CharacterArc:
    character_id: str
    arc_id: str
    status: ArcStatus
    progress: float

    SCHEMA_VERSION: ClassVar[str] = NARRATIVE_SCHEMA_VERSION


@dataclass(frozen=True)
class ResolutionContext:
    """为 Realizer 提供的冲突解决上下文"""
    conflicts: Tuple['Conflict', ...] = field(default_factory=tuple)
    resolutions: Tuple['ConflictResolution', ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        from src.narrative.intent.conflict import Conflict
        from src.narrative.conflict.model import ConflictResolution
        return {
            "conflicts": [c.to_dict() for c in self.conflicts],
            "resolutions": [r.to_dict() for r in self.resolutions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ResolutionContext:
        from src.narrative.intent.conflict import Conflict
        from src.narrative.conflict.model import ConflictResolution
        return cls(
            conflicts=tuple(Conflict.from_dict(c) for c in data.get("conflicts", [])),
            resolutions=tuple(ConflictResolution.from_dict(r) for r in data.get("resolutions", [])),
        )


@dataclass(frozen=True)
class NarrativeContext:
    story: StorySnapshot
    metadata: ChapterMetadata
    previous_chapters: Tuple[str, ...] = field(default_factory=tuple)
    character_arcs: Mapping[str, CharacterArc] = field(default_factory=dict)
    reader_stats: Optional[Mapping[str, Any]] = None
    genre_profile: Optional[Mapping[str, Any]] = None
    context_id: UUID = field(default_factory=uuid4)
    resolution_context: Optional['ResolutionContext'] = None

    SCHEMA_VERSION: ClassVar[str] = NARRATIVE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = {
            "context_id": str(self.context_id),
            "story": self.story.to_dict(),
            "metadata": {
                "volume": self.metadata.volume,
                "chapter": self.metadata.chapter,
                "scene_index": self.metadata.scene_index,
                "total_scenes": self.metadata.total_scenes,
            },
            "previous_chapters": list(self.previous_chapters),
            "character_arcs": {
                k: {
                    "character_id": v.character_id,
                    "arc_id": v.arc_id,
                    "status": v.status.value,
                    "progress": v.progress,
                }
                for k, v in self.character_arcs.items()
            },
            "reader_stats": dict(self.reader_stats) if self.reader_stats else None,
            "genre_profile": dict(self.genre_profile) if self.genre_profile else None,
            "schema_version": self.SCHEMA_VERSION,
        }
        if self.resolution_context:
            data["resolution_context"] = self.resolution_context.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NarrativeContext:
        _ = data.get("schema_version", cls.SCHEMA_VERSION)
        metadata = data.get("metadata", {})
        arcs_data = data.get("character_arcs", {})
        arcs = {
            k: CharacterArc(
                character_id=v["character_id"],
                arc_id=v["arc_id"],
                status=ArcStatus(v["status"]),
                progress=v["progress"],
            )
            for k, v in arcs_data.items()
        }
        resolution_ctx = None
        if "resolution_context" in data:
            # 延迟导入 ResolutionContext，避免循环
            from src.narrative.context import ResolutionContext as RC
            resolution_ctx = RC.from_dict(data["resolution_context"])
        return cls(
            story=StorySnapshot.from_dict(data.get("story", {})),
            metadata=ChapterMetadata(
                volume=metadata.get("volume", 1),
                chapter=metadata.get("chapter", 1),
                scene_index=metadata.get("scene_index", 0),
                total_scenes=metadata.get("total_scenes", 3),
            ),
            previous_chapters=tuple(data.get("previous_chapters", [])),
            character_arcs=arcs,
            reader_stats=data.get("reader_stats"),
            genre_profile=data.get("genre_profile"),
            context_id=parse_uuid(data.get("context_id"), "context_id"),
            resolution_context=resolution_ctx,
        )