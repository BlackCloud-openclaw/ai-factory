# tests/narrative/adaptive/test_router.py

import pytest
from uuid import uuid4
from unittest.mock import MagicMock

from src.narrative.intent import (
    NarrativeIntent,
    IntentSource,
    IntentPriority,
    IntentDimension,
    IntentDirection,
    BuiltinDimensions,
)
from src.narrative.intent.conflict import detect_direction_conflicts, ConflictType, Conflict
from src.narrative.adaptive.router import StrategyProviderRouter
from src.narrative.adaptive.provider import StrategyDecisionProvider
from src.narrative.adaptive.model import StrategyDecision
from src.narrative.conflict import ConflictStrategy


class MockProvider(StrategyDecisionProvider):
    def __init__(self, name: str):
        self.name = name
        self.last_decide = None

    def decide(self, conflicts, intents):
        self.last_decide = ("decide", conflicts, intents)
        return StrategyDecision(
            strategy=ConflictStrategy.BALANCE,
            confidence=1.0,
            reason=f"Mock {self.name}",
            selected_by=self.name,
        )


def create_test_intents_with_novel_id(novel_id: str):
    """创建包含 novel_id 的测试意图，使用 object.__setattr__ 绕过冻结"""
    intent = NarrativeIntent(
        source=IntentSource.SYSTEM,
        dimension=IntentDimension.dialogue(),
        desired_effect="测试意图",
        priority=IntentPriority.MEDIUM,
    )
    # 使用 object.__setattr__ 绕过 frozen 限制（仅用于测试）
    object.__setattr__(intent, "metadata", {"novel_id": novel_id})
    return (intent,)


class TestStrategyProviderRouter:
    def test_zero_rollout_uses_rule(self):
        adaptive = MockProvider("adaptive")
        rule = MockProvider("rule")
        router = StrategyProviderRouter(
            adaptive_provider=adaptive,
            rule_provider=rule,
            rollout_percentage=0,
        )

        intents = create_test_intents_with_novel_id("test_novel_123")
        conflicts = ()

        decision = router.decide(conflicts, intents)

        assert router.last_provider == "rule"
        assert rule.last_decide is not None
        assert adaptive.last_decide is None

    def test_full_rollout_uses_adaptive(self):
        adaptive = MockProvider("adaptive")
        rule = MockProvider("rule")
        router = StrategyProviderRouter(
            adaptive_provider=adaptive,
            rule_provider=rule,
            rollout_percentage=100,
        )

        intents = create_test_intents_with_novel_id("test_novel_123")
        conflicts = ()

        decision = router.decide(conflicts, intents)

        assert router.last_provider == "adaptive"
        assert adaptive.last_decide is not None
        assert rule.last_decide is None

    def test_rollout_bucket_stable(self):
        adaptive = MockProvider("adaptive")
        rule = MockProvider("rule")
        router = StrategyProviderRouter(
            adaptive_provider=adaptive,
            rule_provider=rule,
            rollout_percentage=50,
        )

        intents = create_test_intents_with_novel_id("stable_novel_456")
        conflicts = ()

        router.decide(conflicts, intents)
        bucket1 = router.last_bucket

        router.decide(conflicts, intents)
        bucket2 = router.last_bucket

        assert bucket1 == bucket2

    def test_rollout_boundary_check(self):
        with pytest.raises(ValueError, match="between 0 and 100"):
            StrategyProviderRouter(
                adaptive_provider=MockProvider("adaptive"),
                rule_provider=MockProvider("rule"),
                rollout_percentage=150,
            )

        with pytest.raises(ValueError, match="between 0 and 100"):
            StrategyProviderRouter(
                adaptive_provider=MockProvider("adaptive"),
                rule_provider=MockProvider("rule"),
                rollout_percentage=-10,
            )

    def test_set_rollout_percentage_validates(self):
        router = StrategyProviderRouter(
            adaptive_provider=MockProvider("adaptive"),
            rule_provider=MockProvider("rule"),
            rollout_percentage=0,
        )

        router.set_rollout_percentage(50)
        assert router.rollout_percentage == 50

        with pytest.raises(ValueError):
            router.set_rollout_percentage(101)

        with pytest.raises(ValueError):
            router.set_rollout_percentage(-1)