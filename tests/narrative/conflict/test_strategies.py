# tests/narrative/conflict/test_strategies.py

import pytest
from dataclasses import replace

from src.narrative.intent import (
    NarrativeIntent,
    IntentSource,
    IntentPriority,
    IntentDimension,
    IntentDirection,
    BuiltinDimensions,
)
from src.narrative.intent.conflict import detect_direction_conflicts, ConflictType, Conflict  # ✅ 从正确位置导入
from src.narrative.conflict import (
    PriorityResolver,
    BalanceResolver,
    SynthesisResolver,
    CompositeResolver,
    StrategySelector,
    create_resolver,
    ConflictStrategy,
)


class TestPriorityResolver:
    def test_high_vs_low(self):
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
            desired_effect="减少冗余对白",
            priority=IntentPriority.LOW,
        )
        conflicts = detect_direction_conflicts((inc, dec))
        resolver = PriorityResolver()
        resolutions = resolver.resolve(tuple(conflicts), (inc, dec))

        assert len(resolutions) == 1
        res = resolutions[0]
        assert res.strategy == ConflictStrategy.PRIORITY
        assert res.selected_intent == inc.id

    def test_equal_priority_chooses_first(self):
        inc = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.INCREASE,
            ),
            desired_effect="增加对白",
            priority=IntentPriority.MEDIUM,
        )
        dec = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.DECREASE,
            ),
            desired_effect="减少对白",
            priority=IntentPriority.MEDIUM,
        )
        conflicts = detect_direction_conflicts((inc, dec))
        resolver = PriorityResolver()
        resolutions = resolver.resolve(tuple(conflicts), (inc, dec))

        assert len(resolutions) == 1
        assert resolutions[0].selected_intent == inc.id


class TestBalanceResolver:
    def test_balance_same_priority(self):
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
        conflicts = detect_direction_conflicts((inc, dec))
        resolver = BalanceResolver()
        resolutions = resolver.resolve(tuple(conflicts), (inc, dec))

        assert len(resolutions) == 1
        res = resolutions[0]
        assert res.strategy == ConflictStrategy.BALANCE
        assert res.selected_intent is None
        assert res.chosen_direction is None
        assert "平衡双方目标" in res.rationale

    def test_balance_with_dimension_metadata(self):
        inc = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.EMOTION,
                direction=IntentDirection.INCREASE,
            ),
            desired_effect="增强情感表达",
        )
        dec = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.EMOTION,
                direction=IntentDirection.DECREASE,
            ),
            desired_effect="保持克制",
        )
        conflicts = detect_direction_conflicts((inc, dec))
        resolver = BalanceResolver()
        resolutions = resolver.resolve(tuple(conflicts), (inc, dec))

        assert len(resolutions) == 1
        assert "情感" in resolutions[0].rationale or "表达" in resolutions[0].rationale


class TestSynthesisResolver:
    def test_synthesis_creates_higher_goal(self):
        inc = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.INCREASE,
            ),
            desired_effect="增加角色间互动",
        )
        dec = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.DECREASE,
            ),
            desired_effect="精简叙事节奏",
        )
        conflicts = detect_direction_conflicts((inc, dec))
        resolver = SynthesisResolver()
        resolutions = resolver.resolve(tuple(conflicts), (inc, dec))

        assert len(resolutions) == 1
        res = resolutions[0]
        assert res.strategy == ConflictStrategy.SYNTHESIS
        assert "更高层面" in res.rationale or "统一" in res.rationale


class TestStrategySelector:
    def test_select_priority_when_absolute(self):
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
        original_conflicts = detect_direction_conflicts((inc, dec))
        assert len(original_conflicts) > 0
        conflict = replace(
            original_conflicts[0],
            metadata={
                **original_conflicts[0].metadata,
                "resolution_hint": "absolute"
            }
        )
        selector = StrategySelector()
        resolver = selector.select((conflict,), (inc, dec))
        assert isinstance(resolver, PriorityResolver)

    def test_select_synthesis_when_hint_provided(self):
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
        conflict = Conflict(
            type=ConflictType.DIRECTION_MISMATCH,
            intents=(inc, dec),
            description="合成测试",
            metadata={"dimension": "dialogue", "synthesis_hint": True}
        )
        selector = StrategySelector()
        resolver = selector.select((conflict,), (inc, dec))
        assert isinstance(resolver, SynthesisResolver)

    def test_select_balance_by_default(self):
        inc = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.INCREASE,
            ),
            desired_effect="增加对白",
            priority=IntentPriority.MEDIUM,
        )
        dec = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension(
                id=BuiltinDimensions.DIALOGUE,
                direction=IntentDirection.DECREASE,
            ),
            desired_effect="减少对白",
            priority=IntentPriority.MEDIUM,
        )
        conflict = Conflict(
            type=ConflictType.DIRECTION_MISMATCH,
            intents=(inc, dec),
            description="默认测试",
            metadata={"dimension": "dialogue"}
        )
        selector = StrategySelector()
        resolver = selector.select((conflict,), (inc, dec))
        assert isinstance(resolver, BalanceResolver)


class TestCompositeResolver:
    def test_composite_uses_selector(self):
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
            metadata={"dimension": BuiltinDimensions.DIALOGUE, "resolution_hint": "absolute"}
        )
        resolver = CompositeResolver()
        resolutions = resolver.resolve((conflict,), (inc, dec))
        assert len(resolutions) == 1
        assert resolutions[0].strategy == ConflictStrategy.PRIORITY
        assert resolutions[0].selected_intent == inc.id
        # 新 API: get_last_resolver_name
        assert resolver.get_last_resolver_name() == "PriorityResolver"

    def test_empty_conflicts(self):
        resolver = CompositeResolver()
        result = resolver.resolve((), ())
        assert result == ()
        # 没有冲突时，_last_decision 未设置，get_last_decision 返回 None
        assert resolver.get_last_decision() is None


class TestFactory:
    def test_create_priority(self):
        resolver = create_resolver("priority")
        assert isinstance(resolver, PriorityResolver)

    def test_create_balance(self):
        resolver = create_resolver("balance")
        assert isinstance(resolver, BalanceResolver)

    def test_create_synthesis(self):
        resolver = create_resolver("synthesis")
        assert isinstance(resolver, SynthesisResolver)

    def test_create_composite(self):
        resolver = create_resolver("composite")
        assert isinstance(resolver, CompositeResolver)

    def test_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown conflict strategy"):
            create_resolver("unknown")

    def test_create_default_alias(self):
        from src.narrative.conflict import create_default_resolver
        resolver = create_default_resolver()
        assert isinstance(resolver, PriorityResolver)