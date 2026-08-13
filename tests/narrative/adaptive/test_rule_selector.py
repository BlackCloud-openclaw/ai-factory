# tests/narrative/adaptive/test_rule_selector.py

import pytest
from uuid import uuid4
from dataclasses import replace

from src.narrative.intent import (
    NarrativeIntent,
    IntentSource,
    IntentPriority,
    IntentDimension,
    IntentDirection,
    BuiltinDimensions,
)
from src.narrative.intent.conflict import detect_direction_conflicts, ConflictType, Conflict
from src.narrative.conflict import ConflictStrategy
# ✅ 从 adaptive 包级导入（__init__ 会重新导出）
from src.narrative.adaptive import RuleSelector


class TestRuleSelector:
    def test_no_conflicts_returns_priority(self):
        selector = RuleSelector()
        decision = selector.decide((), ())
        assert decision.strategy == ConflictStrategy.PRIORITY
        assert decision.selected_by == "rule_fallback"

    def test_absolute_conflict_returns_priority(self):
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
        conflict = replace(
            original_conflicts[0],
            metadata={
                **original_conflicts[0].metadata,
                "resolution_hint": "absolute"
            }
        )
        selector = RuleSelector()
        decision = selector.decide((conflict,), (inc, dec))
        assert decision.strategy == ConflictStrategy.PRIORITY

    def test_priority_diff_returns_priority(self):
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
        conflicts = detect_direction_conflicts((inc, dec))
        selector = RuleSelector()
        decision = selector.decide(tuple(conflicts), (inc, dec))
        assert decision.strategy == ConflictStrategy.PRIORITY

    def test_synthesis_hint_returns_synthesis(self):
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
        selector = RuleSelector()
        decision = selector.decide((conflict,), (inc, dec))
        assert decision.strategy == ConflictStrategy.SYNTHESIS

    def test_default_returns_balance(self):
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
        selector = RuleSelector()
        decision = selector.decide(tuple(conflicts), (inc, dec))
        assert decision.strategy == ConflictStrategy.BALANCE