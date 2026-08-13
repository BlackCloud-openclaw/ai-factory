# src/narrative/adaptive/tracker.py

from typing import Optional, List
from uuid import UUID

from src.narrative.conflict import ConflictStrategy
from src.narrative.adaptive.model import StrategyPerformance, StrategyFeedbackEvent
from src.narrative.adaptive.repository import PerformanceRepository, InMemoryRepository


class StrategyPerformanceTracker:
    def __init__(self, repository: Optional[PerformanceRepository] = None):
        self._repo = repository or InMemoryRepository()

    def record(
        self,
        strategy: ConflictStrategy,
        satisfaction: float,
        iterations: int,
        conflict_id: UUID,
        resolution_id: Optional[UUID] = None,
    ) -> None:
        current = self._repo.get_performance(strategy)
        if current is None:
            current = StrategyPerformance(strategy=strategy)
        updated = current.update(satisfaction, iterations)
        self._repo.set_performance(strategy, updated)

        event = StrategyFeedbackEvent(
            conflict_id=conflict_id,
            strategy=strategy,
            satisfaction_score=satisfaction,
            iterations=iterations,
            resolution_id=resolution_id,
        )
        self._repo.save_event(event)

    def get_performance(self, strategy: ConflictStrategy) -> Optional[StrategyPerformance]:
        return self._repo.get_performance(strategy)

    def get_all_performances(self) -> List[StrategyPerformance]:
        return self._repo.get_all_performances()

    def get_events(
        self,
        strategy: Optional[ConflictStrategy] = None,
        limit: int = 20,
    ) -> List[StrategyFeedbackEvent]:
        return self._repo.get_events(strategy=strategy, limit=limit)

    def get_event_count(self, strategy: Optional[ConflictStrategy] = None) -> int:
        return self._repo.get_event_count(strategy=strategy)

    def get_best_performance(self) -> Optional[StrategyPerformance]:
        all_perf = self.get_all_performances()
        if not all_perf:
            return None
        return max(all_perf, key=lambda p: p.avg_satisfaction)