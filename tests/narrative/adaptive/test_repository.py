# tests/narrative/adaptive/test_repository.py

import pytest
from uuid import uuid4

from src.narrative.conflict import ConflictStrategy
from src.narrative.adaptive.model import StrategyPerformance, StrategyFeedbackEvent
from src.narrative.adaptive.repository import InMemoryRepository


class TestInMemoryRepository:
    def test_performance_crud(self):
        repo = InMemoryRepository()
        perf = StrategyPerformance(strategy=ConflictStrategy.PRIORITY)

        repo.set_performance(ConflictStrategy.PRIORITY, perf)
        retrieved = repo.get_performance(ConflictStrategy.PRIORITY)
        assert retrieved == perf

        updated = perf.update(0.8, 1)
        repo.set_performance(ConflictStrategy.PRIORITY, updated)
        retrieved = repo.get_performance(ConflictStrategy.PRIORITY)
        assert retrieved == updated
        assert retrieved.total_uses == 1

        assert repo.get_performance(ConflictStrategy.SYNTHESIS) is None

    def test_get_all_performances(self):
        repo = InMemoryRepository()
        p1 = StrategyPerformance(strategy=ConflictStrategy.PRIORITY)
        p2 = StrategyPerformance(strategy=ConflictStrategy.BALANCE)
        repo.set_performance(ConflictStrategy.PRIORITY, p1)
        repo.set_performance(ConflictStrategy.BALANCE, p2)

        all_perf = repo.get_all_performances()
        assert len(all_perf) == 2

    def test_event_storage(self):
        repo = InMemoryRepository()
        cid = uuid4()
        event = StrategyFeedbackEvent(
            conflict_id=cid,
            strategy=ConflictStrategy.PRIORITY,
            satisfaction_score=0.8,
            iterations=1,
        )
        repo.save_event(event)
        events = repo.get_events()
        assert len(events) == 1
        assert events[0] == event

    def test_event_filter_by_strategy(self):
        repo = InMemoryRepository()
        cid = uuid4()
        e1 = StrategyFeedbackEvent(conflict_id=cid, strategy=ConflictStrategy.PRIORITY, satisfaction_score=0.8, iterations=1)
        e2 = StrategyFeedbackEvent(conflict_id=cid, strategy=ConflictStrategy.BALANCE, satisfaction_score=0.9, iterations=1)
        repo.save_event(e1)
        repo.save_event(e2)

        priority_events = repo.get_events(strategy=ConflictStrategy.PRIORITY)
        assert len(priority_events) == 1
        assert priority_events[0].strategy == ConflictStrategy.PRIORITY

    def test_event_count(self):
        repo = InMemoryRepository()
        cid = uuid4()
        repo.save_event(StrategyFeedbackEvent(conflict_id=cid, strategy=ConflictStrategy.PRIORITY, satisfaction_score=0.8, iterations=1))
        repo.save_event(StrategyFeedbackEvent(conflict_id=cid, strategy=ConflictStrategy.PRIORITY, satisfaction_score=0.9, iterations=1))
        repo.save_event(StrategyFeedbackEvent(conflict_id=cid, strategy=ConflictStrategy.BALANCE, satisfaction_score=0.7, iterations=1))

        assert repo.get_event_count() == 3
        assert repo.get_event_count(strategy=ConflictStrategy.PRIORITY) == 2
        assert repo.get_event_count(strategy=ConflictStrategy.BALANCE) == 1