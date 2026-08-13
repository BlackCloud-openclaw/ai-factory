from dataclasses import dataclass, field
from typing import Any, Optional, Sequence
from .severity import Severity


@dataclass(frozen=True)
class ExpectedStateChange:
    id: str
    field: str
    value: Any
    target: Optional[str] = None
    operation: str = "set"
    description: str = ""
    severity: Severity = Severity.MEDIUM


@dataclass(frozen=True)
class StateExpectation:
    changes: Sequence[ExpectedStateChange] = field(default_factory=tuple)

    @classmethod
    def empty(cls) -> "StateExpectation":
        return cls(changes=())

    def __len__(self) -> int:
        return len(self.changes)

    def __bool__(self) -> bool:
        return len(self.changes) > 0