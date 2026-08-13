# src/writing/artifact/planning.py

from dataclasses import dataclass
from typing import List, Optional, Mapping, Any

JsonValue = Any

@dataclass(frozen=True)
class WorldStateArtifact:
    location: str
    time: str
    weather: str
    realm: Optional[str] = None


@dataclass(frozen=True)
class ConflictArtifact:
    type: str
    description: str
    participants: List[str]
    severity: Optional[str] = None


@dataclass(frozen=True)
class CharacterArtifact:
    id: str
    name: str
    role: str
    realm: Optional[str] = None


@dataclass(frozen=True)
class PlanningCore:
    scene_id: str
    scene_goal: str
    must_events: List[str]
    world_state: WorldStateArtifact
    conflicts: List[ConflictArtifact]
    characters: List[CharacterArtifact]
    emotion_arc: Optional[Mapping[str, str]] = None


@dataclass(frozen=True)
class PlanningArtifact:
    core: PlanningCore
    extension: Mapping[str, Mapping[str, JsonValue]]
    schema_version: str = "1.0"