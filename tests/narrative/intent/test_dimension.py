# tests/narrative/intent/test_dimension.py

import pytest

from src.narrative.intent.dimension import (
    IntentDimension,
    IntentDirection,
    BuiltinDimensions,
)


class TestIntentDimension:
    def test_creation(self):
        dim = IntentDimension(
            id=BuiltinDimensions.DIALOGUE,
            direction=IntentDirection.INCREASE,
        )
        assert dim.id == "narrative.dialogue"
        assert dim.direction == IntentDirection.INCREASE

    def test_opposite_detection(self):
        inc = IntentDimension(
            id=BuiltinDimensions.DIALOGUE,
            direction=IntentDirection.INCREASE,
        )
        dec = IntentDimension(
            id=BuiltinDimensions.DIALOGUE,
            direction=IntentDirection.DECREASE,
        )
        assert inc.is_opposite(dec) is True
        assert dec.is_opposite(inc) is True

    def test_same_direction_not_opposite(self):
        inc1 = IntentDimension(
            id=BuiltinDimensions.DIALOGUE,
            direction=IntentDirection.INCREASE,
        )
        inc2 = IntentDimension(
            id=BuiltinDimensions.DIALOGUE,
            direction=IntentDirection.INCREASE,
        )
        assert inc1.is_opposite(inc2) is False

    def test_transform_not_opposite(self):
        inc = IntentDimension(
            id=BuiltinDimensions.DIALOGUE,
            direction=IntentDirection.INCREASE,
        )
        trans = IntentDimension(
            id=BuiltinDimensions.DIALOGUE,
            direction=IntentDirection.TRANSFORM,
        )
        assert inc.is_opposite(trans) is False
        assert trans.is_opposite(inc) is False

    def test_serialization_roundtrip(self):
        original = IntentDimension(
            id=BuiltinDimensions.EMOTION,
            direction=IntentDirection.TRANSFORM,
            target_value=0.7,
        )
        data = original.to_dict()
        restored = IntentDimension.from_dict(data)
        assert restored.id == original.id
        assert restored.direction == original.direction
        assert restored.target_value == original.target_value