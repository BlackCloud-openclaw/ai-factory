# tests/narrative/test_intent.py

import pytest
from dataclasses import FrozenInstanceError
from uuid import UUID, uuid4

from src.narrative.intent import (
    IntentSource,
    IntentPriority,
    NarrativeIntent,
    NarrativeIntentSet,
    IntentDimension,
    IntentDirection,
    BuiltinDimensions,
)


class TestNarrativeIntent:
    def test_immutable(self):
        intent = NarrativeIntent(
            source=IntentSource.EDITORIAL,
            dimension=IntentDimension.dialogue(),
            desired_effect="test",
        )
        with pytest.raises(FrozenInstanceError):
            intent.desired_effect = "new"  # type: ignore

    def test_preserve_immutable(self):
        intent = NarrativeIntent(
            source=IntentSource.EDITORIAL,
            dimension=IntentDimension.dialogue(),
            desired_effect="test",
            preserve=("a", "b"),
        )
        with pytest.raises(TypeError):
            intent.preserve[0] = "x"  # type: ignore

    def test_avoid_immutable(self):
        intent = NarrativeIntent(
            source=IntentSource.EDITORIAL,
            dimension=IntentDimension.dialogue(),
            desired_effect="test",
            avoid=("x", "y"),
        )
        with pytest.raises(TypeError):
            intent.avoid[0] = "z"  # type: ignore

    def test_required_dimension(self):
        intent = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,  # ✅ 移除 .value
                direction=IntentDirection.INCREASE,
            ),
            desired_effect="test",
        )
        assert intent.dimension.id == BuiltinDimensions.DIALOGUE
        assert intent.dimension.direction == IntentDirection.INCREASE

    def test_serialization_roundtrip(self):
        original = NarrativeIntent(
            source=IntentSource.EDITORIAL,
            dimension=IntentDimension.dialogue(IntentDirection.TRANSFORM),
            desired_effect="Make readers feel tension",
            preserve=("good pacing",),
            avoid=("info-dump",),
            priority=IntentPriority.HIGH,
            rationale="Tension is key",
        )
        data = original.to_dict()
        restored = NarrativeIntent.from_dict(data)

        assert original.id == restored.id
        assert original.desired_effect == restored.desired_effect
        assert original.preserve == restored.preserve
        assert original.avoid == restored.avoid
        assert original.priority == restored.priority
        assert original.rationale == restored.rationale
        assert original.dimension.id == restored.dimension.id
        assert original.dimension.direction == restored.dimension.direction

    def test_uuid_tolerance(self):
        u = uuid4()
        data = {
            "source": "editorial",
            "dimension": {
                "id": BuiltinDimensions.DIALOGUE,  # ✅ 移除 .value
                "direction": "increase",
            },
            "desired_effect": "test",
            "id": str(u),
        }
        intent = NarrativeIntent.from_dict(data)
        assert intent.id == u

    def test_from_dict_missing_id_generates_uuid(self):
        data = {
            "source": "editorial",
            "dimension": {
                "id": BuiltinDimensions.DIALOGUE,  # ✅ 移除 .value
                "direction": "increase",
            },
            "desired_effect": "test",
        }
        intent = NarrativeIntent.from_dict(data)
        assert intent.id is not None
        assert isinstance(intent.id, UUID)
        assert str(intent.id) != "00000000-0000-0000-0000-000000000000"

    def test_schema_version(self):
        assert NarrativeIntent.SCHEMA_VERSION == "1.0.0"


class TestNarrativeIntentSet:
    def test_immutable(self):
        i1 = NarrativeIntent(
            source=IntentSource.EDITORIAL,
            dimension=IntentDimension.dialogue(),
            desired_effect="A",
        )
        set_ = NarrativeIntentSet(intents=(i1,))
        with pytest.raises(TypeError):
            set_.intents[0] = None  # type: ignore

    def test_len_and_bool(self):
        i1 = NarrativeIntent(
            source=IntentSource.EDITORIAL,
            dimension=IntentDimension.dialogue(),
            desired_effect="A",
        )
        i2 = NarrativeIntent(
            source=IntentSource.READER,
            dimension=IntentDimension.emotion(),
            desired_effect="B",
        )
        set_ = NarrativeIntentSet(intents=(i1, i2))
        assert len(set_) == 2
        assert bool(set_) is True

        empty = NarrativeIntentSet()
        assert len(empty) == 0
        assert bool(empty) is False

    def test_iteration(self):
        i1 = NarrativeIntent(
            source=IntentSource.EDITORIAL,
            dimension=IntentDimension.dialogue(),
            desired_effect="A",
        )
        i2 = NarrativeIntent(
            source=IntentSource.READER,
            dimension=IntentDimension.emotion(),
            desired_effect="B",
        )
        set_ = NarrativeIntentSet(intents=(i1, i2))

        count = 0
        for _ in set_:
            count += 1
        assert count == 2

    def test_serialization_roundtrip(self):
        i1 = NarrativeIntent(
            source=IntentSource.EDITORIAL,
            dimension=IntentDimension.dialogue(),
            desired_effect="A",
        )
        i2 = NarrativeIntent(
            source=IntentSource.READER,
            dimension=IntentDimension.emotion(),
            desired_effect="B",
        )
        original = NarrativeIntentSet(intents=(i1, i2))

        data = original.to_dict()
        restored = NarrativeIntentSet.from_dict(data)

        assert len(restored) == 2
        assert restored.intents[0].source == IntentSource.EDITORIAL
        assert restored.intents[1].source == IntentSource.READER
        assert restored.intents[0].dimension.id == BuiltinDimensions.DIALOGUE  # ✅ 移除 .value
        assert restored.intents[1].dimension.id == BuiltinDimensions.EMOTION   # ✅ 移除 .value

    def test_schema_version(self):
        assert NarrativeIntentSet.SCHEMA_VERSION == "1.0.0"


class TestSemanticIntentIntegration:
    def test_dimension_factory_methods(self):
        d1 = IntentDimension.dialogue()
        assert d1.id == BuiltinDimensions.DIALOGUE  # ✅ 移除 .value
        assert d1.direction == IntentDirection.INCREASE

        d2 = IntentDimension.emotion(IntentDirection.DECREASE)
        assert d2.id == BuiltinDimensions.EMOTION   # ✅ 移除 .value
        assert d2.direction == IntentDirection.DECREASE

        d3 = IntentDimension.transition(IntentDirection.STABILIZE)
        assert d3.id == BuiltinDimensions.TRANSITION  # ✅ 移除 .value
        assert d3.direction == IntentDirection.STABILIZE

    def test_dimension_opposite_detection(self):
        inc = IntentDimension.dialogue(IntentDirection.INCREASE)
        dec = IntentDimension.dialogue(IntentDirection.DECREASE)
        trans = IntentDimension.dialogue(IntentDirection.TRANSFORM)

        assert inc.is_opposite(dec) is True
        assert dec.is_opposite(inc) is True
        assert inc.is_opposite(trans) is False
        assert trans.is_opposite(inc) is False

    def test_dimension_serialization(self):
        original = IntentDimension(
            id=BuiltinDimensions.EMOTION,  # ✅ 移除 .value
            direction=IntentDirection.TRANSFORM,
            target_value=0.7,
        )
        data = original.to_dict()
        restored = IntentDimension.from_dict(data)
        assert restored.id == original.id
        assert restored.direction == original.direction
        assert restored.target_value == original.target_value

    def test_intent_with_dimension_in_set(self):
        i1 = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(IntentDirection.INCREASE),
            desired_effect="Increase dialogue",
            priority=IntentPriority.HIGH,
        )
        i2 = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.emotion(IntentDirection.INCREASE),
            desired_effect="Increase emotion",
            priority=IntentPriority.MEDIUM,
        )
        set_ = NarrativeIntentSet(intents=(i1, i2))

        assert len(set_) == 2
        assert set_.intents[0].priority == IntentPriority.HIGH
        assert set_.intents[1].priority == IntentPriority.MEDIUM