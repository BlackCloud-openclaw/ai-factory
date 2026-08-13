# src/narrative/adaptive/repository.py

from typing import Protocol, Optional, List
from uuid import UUID

from src.narrative.conflict import ConflictStrategy
from src.narrative.adaptive.model import StrategyPerformance, StrategyFeedbackEvent


class PerformanceRepository(Protocol):
    def get_performance(self, strategy: ConflictStrategy) -> Optional[StrategyPerformance]:
        ...

    def set_performance(self, strategy: ConflictStrategy, perf: StrategyPerformance) -> None:
        ...

    def get_all_performances(self) -> List[StrategyPerformance]:
        ...

    def save_event(self, event: StrategyFeedbackEvent) -> None:
        ...

    def get_events(
        self,
        strategy: Optional[ConflictStrategy] = None,
        limit: int = 20,
    ) -> List[StrategyFeedbackEvent]:
        ...

    def get_event_count(self, strategy: Optional[ConflictStrategy] = None) -> int:
        ...


class InMemoryRepository:
    def __init__(self):
        self._performances: dict[ConflictStrategy, StrategyPerformance] = {}
        self._events: list[StrategyFeedbackEvent] = []

    def get_performance(self, strategy: ConflictStrategy) -> Optional[StrategyPerformance]:
        return self._performances.get(strategy)

    def set_performance(self, strategy: ConflictStrategy, perf: StrategyPerformance) -> None:
        self._performances[strategy] = perf

    def get_all_performances(self) -> List[StrategyPerformance]:
        return list(self._performances.values())

    def save_event(self, event: StrategyFeedbackEvent) -> None:
        self._events.append(event)

    def get_events(
        self,
        strategy: Optional[ConflictStrategy] = None,
        limit: int = 20,
    ) -> List[StrategyFeedbackEvent]:
        filtered = self._events
        if strategy is not None:
            filtered = [e for e in filtered if e.strategy == strategy]
        return filtered[-limit:]

    def get_event_count(self, strategy: Optional[ConflictStrategy] = None) -> int:
        if strategy is None:
            return len(self._events)
        return sum(1 for e in self._events if e.strategy == strategy)