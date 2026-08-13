# src/writing/coverage/models.py

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Dict

from src.writing.common.severity import Severity


class CoverageStatus(Enum):
    PASS = auto()
    FAIL = auto()
    WEAK = auto()
    PARTIAL = auto()


class CoverageCategory(Enum):
    DIALOGUE_RATIO = auto()
    EMOTION_ARC = auto()
    GROUNDING = auto()
    VOICE = auto()
    LORE = auto()
    STYLE = auto()
    CHARACTER = auto()
    TIMELINE = auto()
    INVENTORY = auto()
    RELATIONSHIP = auto()


@dataclass(frozen=True)
class EvidenceReference:
    paragraph: int
    sentence: int
    text: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None


@dataclass(frozen=True)
class CoverageItem:
    item_id: str
    description: str
    status: CoverageStatus
    score: float
    confidence: float
    evidence: List[EvidenceReference]
    reason: str


@dataclass(frozen=True)
class CoverageFinding:
    severity: Severity
    category: CoverageCategory
    target: str
    current: float
    expected: float
    message: str
    evidence_refs: List[EvidenceReference]


@dataclass(frozen=True)
class CoverageReport:
    overall_score: float
    structural_score: float
    semantic_score: float
    items: List[CoverageItem]
    findings: List[CoverageFinding]
    grounding_breakdown: Dict[str, float]
    schema_version: str = "1.0"