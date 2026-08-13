from dataclasses import dataclass, field
from typing import Any, Optional
from .severity import Severity


@dataclass(frozen=True)
class StateMatchResult:
    field: str
    expectation_id: str
    matched: bool
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    strategy: str = "unknown"
    confidence: float = 1.0
    severity: Severity = Severity.MEDIUM
    details: dict = field(default_factory=dict)