# tests/narrative/intent/test_resolver_integration.py

import pytest

from src.narrative.intent import (
    NarrativeIntent,
    NarrativeIntentSet,
    IntentSource,
    IntentPriority,
    IntentDimension,
    IntentDirection,
    BuiltinDimensions,
    IntentResolver,
    ConflictResolutionError,
)
from src.narrative.conflict import ConflictStrategy


class TestIntentResolverIntegration:
    def test_resolve_with_conflict(self):
        inc = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.INCREASE,
            ),
            desired_effect="增加对白",
            priority=IntentPriority.HIGH,
        )
        dec = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.DECREASE,
            ),
            desired_effect="减少对白",
            priority=IntentPriority.LOW,
        )
        intents = NarrativeIntentSet(intents=(inc, dec))
        resolver = IntentResolver()
        plan = resolver.resolve(intents)

        assert len(plan.conflicts) == 1
        assert len(plan.resolutions) == 1
        res = plan.resolutions[0]
        assert res.strategy == ConflictStrategy.PRIORITY
        assert res.selected_intent == inc.id
        assert res.chosen_direction == IntentDirection.INCREASE

    def test_resolve_no_conflict(self):
        i1 = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(),
            desired_effect="增加对白",
        )
        i2 = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.emotion(),
            desired_effect="增强情绪",
        )
        intents = NarrativeIntentSet(intents=(i1, i2))
        resolver = IntentResolver()
        plan = resolver.resolve(intents)
        assert len(plan.conflicts) == 0
        assert len(plan.resolutions) == 0

    def test_resolve_empty_intents(self):
        intents = NarrativeIntentSet()
        resolver = IntentResolver()
        plan = resolver.resolve(intents)
        assert len(plan.primary_intents) == 0
        assert len(plan.conflicts) == 0
        assert len(plan.resolutions) == 0

    def test_resolution_completeness(self):
        # 使用 DefaultConflictResolver，它应返回与冲突等量的决议
        inc = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.INCREASE,
            ),
            desired_effect="增加",
            priority=IntentPriority.HIGH,
        )
        dec = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.DECREASE,
            ),
            desired_effect="减少",
            priority=IntentPriority.LOW,
        )
        intents = NarrativeIntentSet(intents=(inc, dec))
        resolver = IntentResolver()
        plan = resolver.resolve(intents)
        assert len(plan.conflicts) == len(plan.resolutions)