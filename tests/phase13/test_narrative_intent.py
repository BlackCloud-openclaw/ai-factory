"""
Phase 13.1: NarrativeIntent 数据模型测试 (RC2)
"""

import pytest
from src.writing.narrative_intent import (
    SceneRole,
    NarrativeCondition,
    InteractionPlan,
    NarrativeConsequence,
    NarrativeIntent,
)


def test_scene_role_enum():
    """验证 SceneRole 枚举完整性"""
    roles = [r.value for r in SceneRole]
    expected = [
        "setup", "transition", "discovery", "conflict_escalation",
        "confrontation", "character_decision", "consequence",
        "recovery", "climax_preparation", "climax", "resolution"
    ]
    assert set(roles) == set(expected)
    assert len(roles) == len(expected)


def test_narrative_condition():
    """验证 NarrativeCondition 校验"""
    cond = NarrativeCondition(
        target="knowledge.sect_secret",
        operator="exists",
        expected=True
    )
    assert cond.target == "knowledge.sect_secret"

    with pytest.raises(ValueError):
        NarrativeCondition(
            target="test",
            operator="invalid",
            expected=True
        )


def test_narrative_consequence():
    """验证 NarrativeConsequence"""
    c = NarrativeConsequence(
        target="relationship.trust",
        operation="decrease",
        value=1,
        event_type="relationship_change"
    )
    assert c.target == "relationship.trust"
    assert c.event_type == "relationship_change"

    with pytest.raises(ValueError):
        NarrativeConsequence(
            target="test",
            operation="invalid",
            value=1
        )


def test_interaction_plan():
    """验证 InteractionPlan 基础结构"""
    plan = InteractionPlan(
        participants=["林逸", "师门长老"],
        relationship_changes=["trust -1"],
        conflict="信任危机",
        emotional_shift="从信赖转向怀疑"
    )
    assert plan.participants == ["林逸", "师门长老"]
    assert plan.conflict == "信任危机"


def test_narrative_intent_creation():
    """验证 NarrativeIntent 创建"""
    intent = NarrativeIntent(
        intent_id="test_intent_001",
        scene_role=SceneRole.CONFLICT_ESCALATION,
        objective="让主角意识到师门背后的真实目的",
        preconditions=[
            NarrativeCondition(
                target="knowledge.sect_secret",
                operator="exists",
                expected=True
            )
        ],
        beats=["对峙", "信息揭露", "决定"],
        consequences=[
            NarrativeConsequence(
                target="relationship.trust",
                operation="decrease",
                value=1,
                event_type="relationship_change"
            )
        ]
    )

    assert intent.intent_id == "test_intent_001"
    assert intent.scene_role == SceneRole.CONFLICT_ESCALATION
    assert len(intent.consequences) == 1
    assert intent.consequences[0].operation == "decrease"


def test_missing_scene_role_rejected():
    """验证缺少 scene_role 时拒绝创建"""
    with pytest.raises(ValueError):
        NarrativeIntent(
            intent_id="test",
            objective="test"  # 缺少 scene_role
        )


def test_json_round_trip():
    """验证 JSON 往返序列化"""
    intent = NarrativeIntent(
        intent_id="test_round_trip",
        scene_role=SceneRole.DISCOVERY,
        objective="找到密室钥匙",  # 至少5个字符
        preconditions=[
            NarrativeCondition(
                target="location.密室",
                operator="exists",
                expected=True
            )
        ],
        beats=["搜索", "发现暗格"],
        consequences=[
            NarrativeConsequence(
                target="quest.keys_found",
                operation="set",
                value=True
            )
        ]
    )

    raw = intent.model_dump_json()
    restored = NarrativeIntent.model_validate_json(raw)

    assert restored.intent_id == intent.intent_id
    assert restored.scene_role == intent.scene_role
    assert len(restored.preconditions) == len(intent.preconditions)
    assert restored.preconditions[0].target == intent.preconditions[0].target

def test_intent_id_generation():
    """验证确定性 intent_id 生成"""
    id1 = NarrativeIntent.generate_intent_id(
        scene_id="scene_1_58_0",
        role=SceneRole.CONFLICT_ESCALATION,
        objective="让主角意识到师门背后的真实目的"
    )
    id2 = NarrativeIntent.generate_intent_id(
        scene_id="scene_1_58_0",
        role=SceneRole.CONFLICT_ESCALATION,
        objective="让主角意识到师门背后的真实目的"
    )
    # 相同输入应产生相同 ID
    assert id1 == id2

    id3 = NarrativeIntent.generate_intent_id(
        scene_id="scene_1_58_1",
        role=SceneRole.CONFLICT_ESCALATION,
        objective="让主角意识到师门背后的真实目的"
    )
    # 不同 scene_id 应产生不同 ID
    assert id1 != id3