# src/narrative/intent/legacy.py

from src.narrative.intent.model import (
    NarrativeIntent,
    NarrativeIntentSet,
    IntentSource,
    IntentPriority,
)
from src.narrative.intent.dimension import (
    IntentDimension,
    BuiltinDimensions,
    IntentDirection,
)


class LegacyIntentLoader:
    @classmethod
    def upgrade_intent(cls, old_data: dict) -> NarrativeIntent:
        effect = old_data.get("desired_effect", "")
        dimension = cls._infer_dimension(effect)

        return NarrativeIntent(
            source=IntentSource(old_data.get("source", "system")),
            dimension=dimension,
            desired_effect=effect,
            preserve=tuple(old_data.get("preserve", [])),
            avoid=tuple(old_data.get("avoid", [])),
            priority=IntentPriority(old_data.get("priority", "medium")),
            rationale=old_data.get("rationale", ""),
            id=parse_uuid(old_data.get("id")),
        )

    @classmethod
    def upgrade_set(cls, old_data: dict) -> NarrativeIntentSet:
        intents = old_data.get("intents", [])
        return NarrativeIntentSet(
            intents=tuple(cls.upgrade_intent(i) for i in intents),
        )

    @classmethod
    def _infer_dimension(cls, effect: str) -> IntentDimension:
        effect_lower = effect.lower()
        mapping = {
            "对白": BuiltinDimensions.DIALOGUE,
            "对话": BuiltinDimensions.DIALOGUE,
            "人物互动": BuiltinDimensions.DIALOGUE,
            "情绪": BuiltinDimensions.EMOTION,
            "情感": BuiltinDimensions.EMOTION,
            "心理": BuiltinDimensions.EMOTION,
            "场景": BuiltinDimensions.TRANSITION,
            "过渡": BuiltinDimensions.TRANSITION,
            "衔接": BuiltinDimensions.TRANSITION,
        }

        for keyword, dim_id in mapping.items():
            if keyword in effect_lower:
                return IntentDimension(
                    id=dim_id,
                    direction=IntentDirection.INCREASE,
                )

        return IntentDimension(
            id=BuiltinDimensions.CONTINUITY,
            direction=IntentDirection.STABILIZE,
        )


from src.narrative._utils import parse_uuid