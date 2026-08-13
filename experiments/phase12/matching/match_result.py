from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class MatchResult:
    matched: bool
    confidence: float = 1.0
    strategy: str = "unknown"
    details: dict = field(default_factory=dict)

    @classmethod
    def success(cls, strategy: str, confidence: float = 1.0, details: Optional[dict] = None) -> "MatchResult":
        return cls(matched=True, confidence=confidence, strategy=strategy, details=details or {})

    @classmethod
    def failure(cls, strategy: str = "unknown", details: Optional[dict] = None) -> "MatchResult":
        return cls(matched=False, confidence=0.0, strategy=strategy, details=details or {})