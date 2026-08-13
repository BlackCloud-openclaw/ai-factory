"""
Phase 13.2: NarrativeProjection 数据模型测试
"""

import pytest
from datetime import datetime
from src.writing.narrative_projection import NarrativeProjection
from src.writing.narrative_intent import SceneRole


def test_projection_creation():
    """验证 NarrativeProjection 可正常创建"""
    proj = NarrativeProjection(
        projection_id="test_proj",
        chapter_id="chapter_001",
        active_conflict="师门隐藏真实目的",
        unresolved_threads=["长老真实身份", "血脉来源"],
        active_objectives=["探查秘境"],
        emotional_state="怀疑增强",
        next_pressure="迫使林逸做出选择",
        last_intent_id="intent_001",
        last_scene_role=SceneRole.CONFLICT_ESCALATION,
        version=1,
    )

    assert proj.projection_id == "test_proj"
    assert proj.chapter_id == "chapter_001"
    assert proj.active_conflict == "师门隐藏真实目的"
    assert len(proj.unresolved_threads) == 2
    assert proj.last_scene_role == SceneRole.CONFLICT_ESCALATION
    assert proj.version == 1


def test_projection_version_validation():
    """验证 version 字段校验"""
    with pytest.raises(ValueError):
        NarrativeProjection(
            projection_id="test",
            chapter_id="ch1",
            last_intent_id="intent",
            last_scene_role=SceneRole.SETUP,
            version=0,
        )


def test_projection_to_dict():
    """验证序列化"""
    proj = NarrativeProjection(
        projection_id="test_proj",
        chapter_id="chapter_001",
        active_conflict="test",
        last_intent_id="intent",
        last_scene_role=SceneRole.DISCOVERY,
        version=3,
    )
    data = proj.to_dict()
    assert data["projection_id"] == "test_proj"
    assert data["chapter_id"] == "chapter_001"
    assert data["last_scene_role"] == "discovery"
    assert data["version"] == 3
    assert "updated_at" in data


def test_projection_from_dict():
    """验证反序列化"""
    data = {
        "projection_id": "test_proj",
        "chapter_id": "chapter_001",
        "active_conflict": "测试冲突",
        "unresolved_threads": ["线索1", "线索2"],
        "active_objectives": ["目标1"],
        "emotional_state": "平静",
        "next_pressure": "下一步压力",
        "last_intent_id": "intent_001",
        "last_scene_role": "conflict_escalation",
        "version": 2,
        "updated_at": "2026-07-29T12:00:00",
    }
    proj = NarrativeProjection.from_dict(data)
    assert proj.projection_id == "test_proj"
    assert proj.chapter_id == "chapter_001"
    assert proj.last_scene_role == SceneRole.CONFLICT_ESCALATION
    assert proj.version == 2
    assert isinstance(proj.updated_at, datetime)


def test_projection_id_generation():
    """验证 projection_id 确定性生成"""
    id1 = NarrativeProjection.generate_projection_id(
        chapter_id="chapter_001",
        last_intent_id="intent_001"
    )
    id2 = NarrativeProjection.generate_projection_id(
        chapter_id="chapter_001",
        last_intent_id="intent_001"
    )
    assert id1 == id2

    id3 = NarrativeProjection.generate_projection_id(
        chapter_id="chapter_002",
        last_intent_id="intent_001"
    )
    assert id1 != id3


def test_projection_increment_version():
    """验证版本递增"""
    proj = NarrativeProjection(
        projection_id="test",
        chapter_id="ch1",
        last_intent_id="intent",
        last_scene_role=SceneRole.SETUP,
        version=1,
    )
    new_proj = proj.increment_version()
    assert new_proj.version == 2
    assert new_proj.projection_id == proj.projection_id
    assert new_proj.active_conflict == proj.active_conflict
    # 验证原始对象不变
    assert proj.version == 1