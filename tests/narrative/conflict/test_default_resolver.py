import pytest
from uuid import uuid4

from src.narrative.intent import (
    NarrativeIntent,
    IntentSource,
    IntentPriority,
    IntentDimension,
    IntentDirection,
    BuiltinDimensions,
)
from src.narrative.intent.conflict import Conflict, ConflictType
from src.narrative.conflict import DefaultConflictResolver, ConflictStrategy


class TestDefaultConflictResolver:
    def test_resolve_priority_conflict(self):
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
        # 手动构造冲突，避免依赖 detect_direction_conflicts
        conflict = Conflict(
            type=ConflictType.DIRECTION_MISMATCH,
            intents=(inc, dec),
            description="方向冲突",
        )
        resolver = DefaultConflictResolver()
        resolutions = resolver.resolve((conflict,), (inc, dec))

        assert len(resolutions) == 1
        res = resolutions[0]
        assert res.strategy == ConflictStrategy.PRIORITY
        assert res.selected_intent == inc.id
        assert res.chosen_direction == IntentDirection.INCREASE
        assert "按优先级选择" in res.rationale

    def test_resolve_no_conflict(self):
        resolver = DefaultConflictResolver()
        resolutions = resolver.resolve((), ())
        assert len(resolutions) == 0

    def test_resolve_unknown_intent_fallback(self):
        inc = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(),
            desired_effect="增加",
        )
        dec = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(),
            desired_effect="减少",
        )
        # 手动构造冲突，但传入的 intents 为空
        conflict = Conflict(
            type=ConflictType.DIRECTION_MISMATCH,
            intents=(inc, dec),
            description="方向冲突",
        )
        resolver = DefaultConflictResolver()
        resolutions = resolver.resolve((conflict,), ())  # 传入空意图

        assert len(resolutions) == 1
        res = resolutions[0]
        assert res.strategy == ConflictStrategy.ASK
        assert "无法定位冲突意图" in res.rationale