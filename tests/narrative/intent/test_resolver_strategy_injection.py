# tests/narrative/intent/test_resolver_strategy_injection.py

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
)
from src.narrative.intent.conflict import Conflict, ConflictType  # ✅ 从正确位置导入
from src.narrative.conflict import (
    create_resolver,
    BalanceResolver,
    SynthesisResolver,
    PriorityResolver,
    ConflictStrategy,
    CompositeResolver,
)


class TestIntentResolverWithStrategies:
    def test_resolver_with_balance(self):
        inc = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.INCREASE,
            ),
            desired_effect="增加对话深度",
            priority=IntentPriority.HIGH,
        )
        dec = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.DECREASE,
            ),
            desired_effect="减少无效对白",
            priority=IntentPriority.HIGH,
        )
        intents = NarrativeIntentSet(intents=(inc, dec))
        resolver = IntentResolver(conflict_resolver=create_resolver("balance"))
        plan = resolver.resolve(intents)

        assert len(plan.resolutions) == 1
        assert plan.resolutions[0].strategy == ConflictStrategy.BALANCE
        assert plan.resolutions[0].selected_intent is None

    def test_resolver_with_synthesis(self):
        inc = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.INCREASE,
            ),
            desired_effect="增加互动",
            priority=IntentPriority.MEDIUM,
        )
        dec = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.DECREASE,
            ),
            desired_effect="精简节奏",
            priority=IntentPriority.MEDIUM,
        )
        intents = NarrativeIntentSet(intents=(inc, dec))
        resolver = IntentResolver(conflict_resolver=create_resolver("synthesis"))
        plan = resolver.resolve(intents)

        assert len(plan.resolutions) == 1
        assert plan.resolutions[0].strategy == ConflictStrategy.SYNTHESIS
        assert "更高层面" in plan.resolutions[0].rationale or "统一" in plan.resolutions[0].rationale

    def test_composite_resolver_direct(self):
        """直接测试 CompositeResolver，注入 absolute hint"""
        inc = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.INCREASE,
            ),
            desired_effect="增加关键对白",
            priority=IntentPriority.HIGH,
        )
        dec = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.DECREASE,
            ),
            desired_effect="删除所有对白",
            priority=IntentPriority.LOW,
        )
        conflict = Conflict(
            type=ConflictType.DIRECTION_MISMATCH,
            intents=(inc, dec),
            description="绝对冲突",
            metadata={"dimension": "dialogue", "resolution_hint": "absolute"}
        )
        resolver = CompositeResolver()
        resolutions = resolver.resolve((conflict,), (inc, dec))
        assert len(resolutions) == 1
        assert resolutions[0].strategy == ConflictStrategy.PRIORITY
        assert resolutions[0].selected_intent == inc.id