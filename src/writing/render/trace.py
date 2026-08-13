# src/writing/render/trace.py

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional


class RenderStatus(Enum):
    SUCCESS = auto()
    SKIPPED = auto()
    FAILED = auto()


@dataclass(frozen=True)
class RenderEntry:
    section_id: str
    renderer: str
    version: str
    priority: int
    status: RenderStatus
    chars: int
    estimated_tokens: int
    elapsed_ms: float
    consumed_fields: List[str]
    error: Optional[str] = None


@dataclass(frozen=True)
class RenderTrace:
    entries: List[RenderEntry]
    total_elapsed_ms: float
    schema_version: str = "1.0"