from typing import Protocol, Sequence
from src.writing.planning_contract import ExecutionUnit
from src.writing.events import NarrativeEvent
from .match_result import MatchResult


class ExecutionUnitMatcher(Protocol):
    def covers(self, unit: ExecutionUnit, events: Sequence[NarrativeEvent]) -> MatchResult:
        ...