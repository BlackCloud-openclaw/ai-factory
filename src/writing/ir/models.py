# src/writing/ir/models.py

from dataclasses import dataclass
from typing import List, Optional, Mapping, Any

JsonValue = Any


@dataclass(frozen=True)
class IRDiagnostic:
    severity: str
    field: str
    message: str
    fallback_used: Optional[str] = None


@dataclass(frozen=True)
class WriterIR:
    scene_goal: str
    facts: Mapping[str, JsonValue]
    preferences: Mapping[str, JsonValue]
    constraints: List[str]
    checklist: List[Mapping[str, JsonValue]]
    metadata: Mapping[str, JsonValue]
    schema_version: str = "1.0"