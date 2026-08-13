# tests/narrative/adaptive/test_model.py

import pytest
from uuid import uuid4

from src.narrative.conflict import ConflictStrategy
from src.narrative.adaptive.model import (
    StrategyPerformance,
    StrategyFeedbackEvent,
    StrategyDecision,
    SelectionMode,
)


class TestStrategyPerformance:
    def test_update_creates_new_instance(self):
        perf = StrategyPerformance(strategy=ConflictStrategy.PRIORITY)
        updated = perf.update(satisfaction=0.8, iterations=2)

        assert updated.total_uses == 1
        assert updated.total_satisfaction == 0.8
        assert updated.avg_satisfaction == 0.8
        assert updated.total_iterations == 2
        assert updated.avg_iterations == 2.0
        assert updated.last_used is not None
        assert updated is not perf

    def test_multiple_updates(self):
        perf = StrategyPerformance(strategy=ConflictStrategy.BALANCE)
        perf = perf.update(0.9, 1)
        perf = perf.update(0.7, 2)

        assert perf.total_uses == 2
        assert perf.total_satisfaction == 1.6
        assert perf.avg_satisfaction == 0.8
        assert perf.total_iterations == 3
        assert perf.avg_iterations == 1.5

    def test_validation_satisfaction_bounds(self):
        perf = StrategyPerformance(strategy=ConflictStrategy.PRIORITY)
        with pytest.raises(ValueError):
            perf.update(1.5, 1)
        with pytest.raises(ValueError):
            perf.update(-0.1, 1)

    def test_validation_iterations_negative(self):
        perf = StrategyPerformance(strategy=ConflictStrategy.PRIORITY)
        with pytest.raises(ValueError):
            perf.update(0.5, -1)


class TestStrategyFeedbackEvent:
    def test_creation(self):
        cid = uuid4()
        rid = uuid4()
        event = StrategyFeedbackEvent(
            conflict_id=cid,
            strategy=ConflictStrategy.PRIORITY,
            satisfaction_score=0.85,
            iterations=1,
            resolution_id=rid,
        )
        assert event.conflict_id == cid
        assert event.strategy == ConflictStrategy.PRIORITY
        assert event.satisfaction_score == 0.85
        assert event.iterations == 1
        assert event.resolution_id == rid
        assert event.event_id is not None
        assert event.timestamp is not None


class TestStrategyDecision:
    def test_creation(self):
        decision = StrategyDecision(
            strategy=ConflictStrategy.BALANCE,
            confidence=0.9,
            reason="测试",
            selected_by="rule",
            historical_score=0.85,
        )
        assert decision.strategy == ConflictStrategy.BALANCE
        assert decision.confidence == 0.9
        assert decision.reason == "测试"
        assert decision.selected_by == "rule"
        assert decision.historical_score == 0.85