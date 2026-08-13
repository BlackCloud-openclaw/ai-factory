"""
Phase 13.2: Projection Initialization 测试

验证 ADR-035 Addendum：
- 首次生成时 previous=None 允许
- version 从 1 开始
- projection_id 确定性生成
"""

import pytest
from src.writing.narrative_projection import NarrativeProjection
from src.writing.narrative_intent import NarrativeIntent, SceneRole
from src.writing.projection_updater import ProjectionUpdater


def test_first_chapter_creates_projection():
    """测试首次生成时创建初始 Projection"""
    updater = ProjectionUpdater()
    intent = NarrativeIntent(
        intent_id="intent_first",
        scene_role=SceneRole.CONFLICT_ESCALATION,
        objective="开启师门秘密调查",
        consequences=[]
    )
    proj = updater.update(None, intent, [])
    assert proj is not None
    assert proj.version == 1
    assert proj.last_intent_id == "intent_first"
    assert proj.chapter_id == "initial"  # 初始状态标记


def test_projection_id_deterministic():
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


def test_first_chapter_active_conflict_inference():
    """测试首次生成时 active_conflict 的推断"""
    updater = ProjectionUpdater()

    # 冲突类角色 → objective 作为 conflict
    intent_conflict = NarrativeIntent(
        intent_id="intent_conflict",
        scene_role=SceneRole.CONFLICT_ESCALATION,
        objective="门派内部冲突",
        consequences=[]
    )
    proj = updater.update(None, intent_conflict, [])
    assert proj.active_conflict == "门派内部冲突"

    # 非冲突角色 → conflict 为 None
    intent_discovery = NarrativeIntent(
        intent_id="intent_discovery",
        scene_role=SceneRole.DISCOVERY,
        objective="发现隐藏密室",
        consequences=[]
    )
    proj = updater.update(None, intent_discovery, [])
    assert proj.active_conflict is None


def test_first_chapter_active_objectives():
    """测试首次生成时 active_objectives 初始化"""
    updater = ProjectionUpdater()
    intent = NarrativeIntent(
        intent_id="intent_obj",
        scene_role=SceneRole.SETUP,
        objective="建立故事基调",
        consequences=[]
    )
    proj = updater.update(None, intent, [])
    assert "建立故事基调" in proj.active_objectives


def test_projection_version_monotonic():
    """验证版本单调递增"""
    updater = ProjectionUpdater()
    intent = NarrativeIntent(
        intent_id="intent_v1",
        scene_role=SceneRole.SETUP,
        objective="设置初始场景",  # 至少5个字符
        consequences=[]
    )
    proj1 = updater.update(None, intent, [])
    assert proj1.version == 1

    proj2 = updater.update(proj1, intent, [])
    assert proj2.version == 2

    proj3 = updater.update(proj2, intent, [])
    assert proj3.version == 3