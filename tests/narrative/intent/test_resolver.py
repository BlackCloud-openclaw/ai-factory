# tests/narrative/intent/test_resolver.py

import pytest

from src.narrative.intent import (
    NarrativeIntent,
    NarrativeIntentSet,
    IntentSource,
    IntentPriority,
    IntentDimension,
    IntentDirection,
    BuiltinDimensions,
    ConflictType,
    ResolutionStrategy,
    IntentResolver,
)


class TestIntentResolver:
    def test_priority_ordering(self):
        low = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.INCREASE,
            ),
            desired_effect="Low priority",
            priority=IntentPriority.LOW,
        )
        high = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.INCREASE,
            ),
            desired_effect="High priority",
            priority=IntentPriority.HIGH,
        )
        intents = NarrativeIntentSet(intents=(low, high))
        resolver = IntentResolver()
        plan = resolver.resolve(intents)

        assert len(plan.primary_intents) == 2
        assert plan.primary_intents[0].priority == IntentPriority.HIGH
        assert plan.primary_intents[1].priority == IntentPriority.LOW

    def test_empty_intents(self):
        intents = NarrativeIntentSet()
        resolver = IntentResolver()
        plan = resolver.resolve(intents)

        assert len(plan.primary_intents) == 0
        assert len(plan.conflicts) == 0
        assert plan.metadata.get("reason") == "no_intents"

    def test_direction_conflict(self):
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
            priority=IntentPriority.HIGH,
        )
        intents = NarrativeIntentSet(intents=(inc, dec))
        resolver = IntentResolver()
        plan = resolver.resolve(intents)

        assert len(plan.conflicts) == 1
        assert plan.conflicts[0].type == ConflictType.DIRECTION_MISMATCH
        assert len(plan.conflicts[0].intents) == 2

    def test_transform_not_conflict_with_increase(self):
        inc = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.INCREASE,
            ),
            desired_effect="增加对白量",
        )
        trans = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.TRANSFORM,
            ),
            desired_effect="将说明对白转为冲突对白",
        )
        intents = NarrativeIntentSet(intents=(inc, trans))
        resolver = IntentResolver()
        plan = resolver.resolve(intents)

        assert len(plan.conflicts) == 0

    def test_different_dimensions_no_conflict(self):
        dialogue = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.INCREASE,
            ),
            desired_effect="增加对白",
        )
        emotion = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.EMOTION,
                direction=IntentDirection.INCREASE,
            ),
            desired_effect="增强情绪",
        )
        intents = NarrativeIntentSet(intents=(dialogue, emotion))
        resolver = IntentResolver()
        plan = resolver.resolve(intents)

        assert len(plan.conflicts) == 0

    def test_resolution_plan_metadata(self):
        inc = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.INCREASE,
            ),
            desired_effect="增加对白",
        )
        dec = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.DECREASE,
            ),
            desired_effect="减少对白",
        )
        intents = NarrativeIntentSet(intents=(inc, dec))
        resolver = IntentResolver()
        plan = resolver.resolve(intents)

        assert plan.metadata.get("total_intents") == 2
        assert plan.metadata.get("conflict_count") == 1
        assert plan.strategy == ResolutionStrategy.PRIORITY_BASED

    def test_resolution_plan_serialization(self):
        inc = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.INCREASE,
            ),
            desired_effect="增加对白",
        )
        intents = NarrativeIntentSet(intents=(inc,))
        resolver = IntentResolver()
        plan = resolver.resolve(intents)

        data = plan.to_dict()
        assert "primary_intents" in data
        assert "conflicts" in data
        assert "strategy" in data
        assert "metadata" in data
        assert len(data["primary_intents"]) == 1
        assert data["primary_intents"][0]["dimension"]["id"] == BuiltinDimensions.DIALOGUE