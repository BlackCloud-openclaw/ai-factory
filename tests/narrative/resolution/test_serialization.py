# tests/narrative/resolution/test_serialization.py

import pytest
from uuid import uuid4

from src.narrative.context import ResolutionContext
from src.narrative.intent.conflict import Conflict, ConflictType
from src.narrative.conflict import ConflictResolution, ConflictStrategy
from src.narrative.intent import NarrativeIntent, IntentSource, IntentDimension, IntentDirection, BuiltinDimensions


def test_resolution_context_roundtrip():
    intent_a = NarrativeIntent(
        source=IntentSource.SYSTEM,
        dimension=IntentDimension.dialogue(),
        desired_effect="增加对白",
    )
    intent_b = NarrativeIntent(
        source=IntentSource.SYSTEM,
        dimension=IntentDimension.dialogue(),
        desired_effect="减少对白",
    )
    conflict = Conflict(
        type=ConflictType.DIRECTION_MISMATCH,
        intents=(intent_a, intent_b),
        description="测试冲突",
    )
    resolution = ConflictResolution(
        conflict_id=conflict.id,
        strategy=ConflictStrategy.PRIORITY,
        rationale="按优先级选择",
        selected_intent=intent_a.id,
    )
    ctx = ResolutionContext(
        conflicts=(conflict,),
        resolutions=(resolution,),
    )

    data = ctx.to_dict()
    restored = ResolutionContext.from_dict(data)

    assert len(restored.conflicts) == 1
    assert len(restored.resolutions) == 1
    assert restored.conflicts[0].id == conflict.id
    assert restored.conflicts[0].type == conflict.type
    assert restored.resolutions[0].conflict_id == resolution.conflict_id
    assert restored.resolutions[0].strategy == resolution.strategy
    assert restored.resolutions[0].selected_intent == resolution.selected_intent


def test_conflict_roundtrip_intents_empty():
    """验证 Conflict 序列化/反序列化后 intents 为空（设计行为）"""
    intent_a = NarrativeIntent(
        source=IntentSource.SYSTEM,
        dimension=IntentDimension.dialogue(),
        desired_effect="增加对白",
    )
    intent_b = NarrativeIntent(
        source=IntentSource.SYSTEM,
        dimension=IntentDimension.dialogue(),
        desired_effect="减少对白",
    )
    conflict = Conflict(
        type=ConflictType.DIRECTION_MISMATCH,
        intents=(intent_a, intent_b),
        description="测试冲突",
    )
    data = conflict.to_dict()
    restored = Conflict.from_dict(data)

    assert restored.intents == ()
    assert restored.id == conflict.id
    assert restored.type == conflict.type
    assert restored.description == conflict.description