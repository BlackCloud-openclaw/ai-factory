# src/narrative/adaptive/feedback.py

from typing import Tuple
from uuid import UUID
from src.narrative.conflict.model import ConflictResolution
from src.narrative.adaptive.tracker import StrategyPerformanceTracker


class StrategyFeedbackCollector:
    def __init__(self, tracker: StrategyPerformanceTracker):
        self._tracker = tracker

    def collect_from_resolutions(
        self,
        resolutions: Tuple[ConflictResolution, ...],
        satisfaction: float,
        iterations: int,
    ) -> None:
        for res in resolutions:
            self._tracker.record(
                strategy=res.strategy,
                satisfaction=satisfaction,
                iterations=iterations,
                conflict_id=res.conflict_id,
                resolution_id=res.resolution_id,
            )