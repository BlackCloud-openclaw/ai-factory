# tests/narrative/adaptive/test_tracker.py

import pytest
from uuid import uuid4

from src.narrative.conflict import ConflictStrategy
from src.narrative.adaptive.tracker import StrategyPerformanceTracker


class TestStrategyPerformanceTracker:
    def test_record_updates_performance(self):
        tracker = StrategyPerformanceTracker()
        cid = uuid4()
        rid = uuid4()

        tracker.record(ConflictStrategy.PRIORITY, 0.8, 1, cid, rid)

        perf = tracker.get_performance(ConflictStrategy.PRIORITY)
        assert perf is not None
        assert perf.total_uses == 1
        assert perf.avg_satisfaction == 0.8
        assert perf.total_iterations == 1

        events = tracker.get_events(strategy=ConflictStrategy.PRIORITY)
        assert len(events) == 1
        assert events[0].conflict_id == cid
        assert events[0].resolution_id == rid
        assert events[0].satisfaction_score == 0.8
        assert events[0].iterations == 1

    def test_multiple_records(self):
        tracker = StrategyPerformanceTracker()
        cid = uuid4()

        tracker.record(ConflictStrategy.BALANCE, 0.9, 1, cid)
        tracker.record(ConflictStrategy.BALANCE, 0.7, 2, cid)

        perf = tracker.get_performance(ConflictStrategy.BALANCE)
        assert perf.total_uses == 2
        assert perf.total_satisfaction == 1.6
        assert perf.avg_satisfaction == 0.8
        assert perf.total_iterations == 3
        assert perf.avg_iterations == 1.5

    def test_get_all_performances(self):
        tracker = StrategyPerformanceTracker()
        cid = uuid4()

        tracker.record(ConflictStrategy.PRIORITY, 0.8, 1, cid)
        tracker.record(ConflictStrategy.BALANCE, 0.9, 1, cid)

        all_perf = tracker.get_all_performances()
        assert len(all_perf) == 2

    def test_get_best_performance(self):
        tracker = StrategyPerformanceTracker()
        cid = uuid4()

        tracker.record(ConflictStrategy.PRIORITY, 0.6, 1, cid)
        tracker.record(ConflictStrategy.BALANCE, 0.9, 1, cid)
        tracker.record(ConflictStrategy.BALANCE, 0.8, 1, cid)

        best = tracker.get_best_performance()
        assert best is not None
        assert best.strategy == ConflictStrategy.BALANCE
        # 使用 approx 处理浮点误差
        assert best.avg_satisfaction == pytest.approx(0.85)

    def test_event_count(self):
        tracker = StrategyPerformanceTracker()
        cid = uuid4()

        tracker.record(ConflictStrategy.PRIORITY, 0.8, 1, cid)
        tracker.record(ConflictStrategy.PRIORITY, 0.9, 1, cid)
        tracker.record(ConflictStrategy.BALANCE, 0.7, 1, cid)

        assert tracker.get_event_count() == 3
        assert tracker.get_event_count(strategy=ConflictStrategy.PRIORITY) == 2
        assert tracker.get_event_count(strategy=ConflictStrategy.BALANCE) == 1