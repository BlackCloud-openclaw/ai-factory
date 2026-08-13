# tests/narrative/conflict/test_composite_adaptive.py

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
from src.narrative.intent.conflict import ConflictType, Conflict
from src.narrative.conflict import (
    CompositeResolver,
    ConflictStrategy,
)
from src.narrative.adaptive import (
    create_adaptive_resolver,
    create_deterministic_resolver,
    InMemoryRepository,
    StrategyPerformanceTracker,
    AdaptiveSelector,
    SelectionMode,
)


class TestCompositeAdaptive:
    def test_composite_default_uses_rule_selector(self):
        # 手动构造冲突，确保检测到
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
        # 直接构造冲突
        conflict = Conflict(
            type=ConflictType.DIRECTION_MISMATCH,
            intents=(inc, dec),
            description="测试冲突",
            metadata={"dimension": BuiltinDimensions.DIALOGUE}
        )
        resolver = CompositeResolver()
        resolutions = resolver.resolve((conflict,), (inc, dec))
        assert len(resolutions) == 1
        # 默认规则策略应选 PRIORITY（因为优先级差）
        assert resolutions[0].strategy == ConflictStrategy.PRIORITY

    def test_composite_with_adaptive_selector(self):
        repo = InMemoryRepository()
        tracker = StrategyPerformanceTracker(repo)
        cid = uuid4()

        # 让 BALANCE 历史表现好，PRIORITY 表现差
        for _ in range(5):
            tracker.record(ConflictStrategy.BALANCE, 0.9, 1, cid)
            tracker.record(ConflictStrategy.PRIORITY, 0.5, 1, cid)

        selector = AdaptiveSelector(
            tracker,
            mode=SelectionMode.ADAPTIVE,
            min_records_for_adaptive=3,
            confidence_threshold=0.05,
        )
        resolver = CompositeResolver(provider=selector)

        # 创建 HIGH vs LOW 冲突，规则策略应为 PRIORITY
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
        # 手动构造冲突
        conflict = Conflict(
            type=ConflictType.DIRECTION_MISMATCH,
            intents=(inc, dec),
            description="测试",
            metadata={"dimension": BuiltinDimensions.DIALOGUE}
        )
        resolutions = resolver.resolve((conflict,), (inc, dec))
        assert len(resolutions) == 1
        # 自适应应选择 BALANCE（历史更好），而非规则 PRIORITY
        assert resolutions[0].strategy == ConflictStrategy.BALANCE
        decision = resolver.get_last_decision()
        assert decision is not None
        # 应为 adaptive 选择
        assert decision.selected_by == "adaptive"

    def test_create_adaptive_resolver_factory(self):
        resolver = create_adaptive_resolver(mode=SelectionMode.ADAPTIVE)
        assert isinstance(resolver, CompositeResolver)

        inc = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(),
            desired_effect="增加对白",
            priority=IntentPriority.MEDIUM,
        )
        dec = NarrativeIntent(
            source=IntentSource.SYSTEM,
            dimension=IntentDimension.dialogue(),
            desired_effect="减少对白",
            priority=IntentPriority.MEDIUM,
        )
        conflict = Conflict(
            type=ConflictType.DIRECTION_MISMATCH,
            intents=(inc, dec),
            description="测试",
            metadata={"dimension": BuiltinDimensions.DIALOGUE}
        )
        resolutions = resolver.resolve((conflict,), (inc, dec))
        assert len(resolutions) == 1
        # 默认规则策略（无优先级差，无 hint）→ BALANCE
        assert resolutions[0].strategy == ConflictStrategy.BALANCE

    def test_create_deterministic_resolver(self):
        resolver = create_deterministic_resolver()

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
        r1 = resolver.resolve((conflict,), (inc, dec))
        r2 = resolver.resolve((conflict,), (inc, dec))
        assert len(r1) == 1
        assert len(r2) == 1
        assert r1[0].strategy == r2[0].strategy
        assert r1[0].strategy == ConflictStrategy.PRIORITY  # 因为绝对冲突