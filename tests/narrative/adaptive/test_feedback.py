# tests/narrative/adaptive/test_feedback.py

import pytest
from uuid import uuid4

from src.narrative.conflict import ConflictStrategy
from src.narrative.conflict.model import ConflictResolution
from src.narrative.adaptive import StrategyFeedbackCollector, StrategyPerformanceTracker, InMemoryRepository


class TestStrategyFeedbackCollector:
    def test_collect_from_resolutions(self):
        repo = InMemoryRepository()
        tracker = StrategyPerformanceTracker(repo)
        collector = StrategyFeedbackCollector(tracker)

        cid = uuid4()
        rid = uuid4()
        res = ConflictResolution(
            conflict_id=cid,
            strategy=ConflictStrategy.BALANCE,
            rationale="测试",
            resolution_id=rid,
        )
        collector.collect_from_resolutions((res,), satisfaction=0.85, iterations=2)

        perf = tracker.get_performance(ConflictStrategy.BALANCE)
        assert perf is not None
        assert perf.total_uses == 1
        assert perf.avg_satisfaction == 0.85
        assert perf.total_iterations == 2

        events = tracker.get_events(strategy=ConflictStrategy.BALANCE)
        assert len(events) == 1
        assert events[0].conflict_id == cid
        assert events[0].resolution_id == rid
        assert events[0].satisfaction_score == 0.85
        assert events[0].iterations == 2

    def test_collect_multiple_resolutions(self):
        repo = InMemoryRepository()
        tracker = StrategyPerformanceTracker(repo)
        collector = StrategyFeedbackCollector(tracker)

        res1 = ConflictResolution(conflict_id=uuid4(), strategy=ConflictStrategy.PRIORITY, rationale="")
        res2 = ConflictResolution(conflict_id=uuid4(), strategy=ConflictStrategy.BALANCE, rationale="")
        collector.collect_from_resolutions((res1, res2), satisfaction=0.75, iterations=3)

        perf_priority = tracker.get_performance(ConflictStrategy.PRIORITY)
        perf_balance = tracker.get_performance(ConflictStrategy.BALANCE)
        assert perf_priority is not None
        assert perf_balance is not None
        assert perf_priority.total_uses == 1
        assert perf_balance.total_uses == 1
        assert perf_priority.avg_satisfaction == 0.75
        assert perf_balance.avg_satisfaction == 0.75