from typing import Protocol
from ..state_expectation import ExpectedStateChange
from ..state_match_result import StateMatchResult
from ..snapshot_accessor import SnapshotAccessor


class StateFieldComparator(Protocol):
    @property
    def field(self) -> str: ...

    def compare(
        self,
        expected: ExpectedStateChange,
        before: SnapshotAccessor,
        after: SnapshotAccessor,
    ) -> StateMatchResult:
        ...