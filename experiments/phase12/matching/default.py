from typing import Dict, Optional, Sequence
from src.writing.planning_contract import ExecutionUnit
from src.writing.events import NarrativeEvent, EventType
from .protocol import ExecutionUnitMatcher
from .match_result import MatchResult
from .mappings import DEFAULT_KEYWORD_MAPPING


class RuleExecutionUnitMatcher(ExecutionUnitMatcher):
    def __init__(
        self,
        keyword_mapping: Optional[Dict[str, EventType]] = None,
    ):
        self._keyword_mapping = keyword_mapping or DEFAULT_KEYWORD_MAPPING

    def covers(self, unit: ExecutionUnit, events: Sequence[NarrativeEvent]) -> MatchResult:
        description = unit.description

        matched_types = set()
        for keyword, event_type in self._keyword_mapping.items():
            if keyword in description:
                for event in events:
                    if event.type == event_type:
                        matched_types.add(event_type.value)
                        break

        if matched_types:
            return MatchResult.success(
                strategy="keyword_match",
                confidence=0.8,
                details={"matched_types": list(matched_types), "description": description[:50]}
            )

        return MatchResult.failure(
            strategy="no_match",
            details={"description": description[:50], "event_types": [e.type.value for e in events[:3]]}
        )