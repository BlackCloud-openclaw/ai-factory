# src/narrative/intent/dimension.py

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional


class IntentDirection(StrEnum):
    INCREASE = "increase"
    DECREASE = "decrease"
    STABILIZE = "stabilize"
    TRANSFORM = "transform"


class BuiltinDimensions:
    DIALOGUE = "narrative.dialogue"
    EMOTION = "narrative.emotion"
    TRANSITION = "narrative.transition"
    PACING = "narrative.pacing"
    VOICE = "narrative.voice"
    HOOK = "narrative.hook"
    CONTINUITY = "narrative.continuity"


@dataclass(frozen=True)
class IntentDimension:
    id: str
    direction: IntentDirection
    target_value: Optional[float] = None

    SCHEMA_VERSION: str = "1.0.0"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "direction": self.direction.value,
            "target_value": self.target_value,
            "schema_version": self.SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IntentDimension":
        return cls(
            id=data["id"],
            direction=IntentDirection(data["direction"]),
            target_value=data.get("target_value"),
        )

    def is_opposite(self, other: "IntentDimension") -> bool:
        if self.id != other.id:
            return False
        if self.direction == other.direction:
            return False
        if {self.direction, other.direction} == {IntentDirection.INCREASE, IntentDirection.DECREASE}:
            return True
        return False

    @classmethod
    def dialogue(cls, direction: IntentDirection = IntentDirection.INCREASE) -> "IntentDimension":
        return cls(id=BuiltinDimensions.DIALOGUE, direction=direction)

    @classmethod
    def emotion(cls, direction: IntentDirection = IntentDirection.INCREASE) -> "IntentDimension":
        return cls(id=BuiltinDimensions.EMOTION, direction=direction)

    @classmethod
    def transition(cls, direction: IntentDirection = IntentDirection.INCREASE) -> "IntentDimension":
        return cls(id=BuiltinDimensions.TRANSITION, direction=direction)