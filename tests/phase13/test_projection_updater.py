"""
Phase 13.2: ProjectionUpdater 测试
"""

import pytest
from src.writing.narrative_projection import NarrativeProjection
from src.writing.narrative_intent import (
    NarrativeIntent, SceneRole, NarrativeConsequence
)
from src.writing.events import (
    DiscoveryEvent,
    PlotFlagSetEvent,
    RelationshipChangeEvent,
)
from src.writing.projection_updater import ProjectionUpdater


def test_updater_initial():
    """测试初始状态创建"""
    updater = ProjectionUpdater()
    intent = NarrativeIntent(
        intent_id="intent_001",
        scene_role=SceneRole.CONFLICT_ESCALATION,
        objective="揭露师门隐藏的秘密",
        consequences=[
            NarrativeConsequence(
                target="knowledge.sect_secret",
                operation="set",
                value=True,
                event_type="plot_flag_set"
            )
        ]
    )
    proj = updater.update(None, intent, [])
    assert proj is not None
    assert proj.active_conflict == "揭露师门隐藏的秘密"
    assert proj.version == 1


def test_updater_initial_no_conflict_role():
    """测试非冲突角色的初始冲突为 None"""
    updater = ProjectionUpdater()
    intent = NarrativeIntent(
        intent_id="intent_001",
        scene_role=SceneRole.DISCOVERY,
        objective="寻找师门线索",
        consequences=[]
    )
    proj = updater.update(None, intent, [])
    assert proj.active_conflict is None
    assert proj.version == 1


def test_updater_threads():
    """测试线程更新"""
    updater = ProjectionUpdater()
    intent = NarrativeIntent(
        intent_id="intent_001",
        scene_role=SceneRole.DISCOVERY,
        objective="发现长老身份线索",
        consequences=[
            NarrativeConsequence(
                target="knowledge.elder_identity",
                operation="set",
                value=True
            )
        ]
    )
    events = [
        DiscoveryEvent(
            discoverer="林逸",
            discovery="发现长老隐藏的身份",
            importance="high"
        )
    ]
    previous = NarrativeProjection(
        projection_id="test",
        chapter_id="ch1",
        active_conflict="师门秘密",
        unresolved_threads=["old_thread"],
        last_intent_id="intent_prev",
        last_scene_role=SceneRole.SETUP,
        version=1,
    )
    proj = updater.update(previous, intent, events)
    assert len(proj.unresolved_threads) == 2
    assert "elder_identity" in proj.unresolved_threads
    assert "old_thread" in proj.unresolved_threads
    assert proj.version == 2


def test_updater_resolve_thread():
    """测试解决线程"""
    updater = ProjectionUpdater()
    previous = NarrativeProjection(
        projection_id="test",
        chapter_id="ch1",
        active_conflict="师门秘密",
        unresolved_threads=["elder_identity", "old_thread"],
        last_intent_id="intent_prev",
        last_scene_role=SceneRole.SETUP,
        version=1,
    )
    intent = NarrativeIntent(
        intent_id="intent_002",
        scene_role=SceneRole.DISCOVERY,
        objective="查明长老真实身份",
        consequences=[]
    )
    events = [
        DiscoveryEvent(
            discoverer="林逸",
            discovery="elder_identity 已查清",
            importance="critical"
        ),
        PlotFlagSetEvent(
            flag="solved_elder_identity",
            value=True
        )
    ]
    proj = updater.update(previous, intent, events)
    assert len(proj.unresolved_threads) == 1
    assert "elder_identity" not in proj.unresolved_threads
    assert "old_thread" in proj.unresolved_threads


def test_updater_active_conflict():
    """测试冲突升级"""
    updater = ProjectionUpdater()
    previous = NarrativeProjection(
        projection_id="test",
        chapter_id="ch1",
        active_conflict="初始冲突",
        unresolved_threads=[],
        last_intent_id="intent_prev",
        last_scene_role=SceneRole.SETUP,
        version=1,
    )
    intent = NarrativeIntent(
        intent_id="intent_003",
        scene_role=SceneRole.CONFLICT_ESCALATION,
        objective="冲突升级：师门对峙",
        consequences=[]
    )
    events = [
        RelationshipChangeEvent(
            from_char="林逸",
            to_char="长老",
            delta=-10,
            new_value=0
        )
    ]
    proj = updater.update(previous, intent, events)
    assert proj.active_conflict == "初始冲突 → 冲突升级：师门对峙"
    assert proj.version == 2


def test_updater_next_pressure():
    """测试 next_pressure 生成"""
    updater = ProjectionUpdater()
    previous = NarrativeProjection(
        projection_id="test",
        chapter_id="ch1",
        active_conflict="师门秘密",
        unresolved_threads=["长老身份", "血脉来源"],
        last_intent_id="intent_prev",
        last_scene_role=SceneRole.SETUP,
        version=1,
    )
    intent = NarrativeIntent(
        intent_id="intent_004",
        scene_role=SceneRole.DISCOVERY,
        objective="调查血脉来源",
        consequences=[]
    )
    proj = updater.update(previous, intent, [])
    assert "长老身份" in proj.next_pressure
    assert "血脉来源" in proj.next_pressure


def test_update_is_immutable():
    """验证 ProjectionUpdater 不修改输入对象"""
    updater = ProjectionUpdater()
    previous = NarrativeProjection(
        projection_id="test",
        chapter_id="ch1",
        active_conflict="原始冲突",
        unresolved_threads=["thread_a"],
        last_intent_id="intent_prev",
        last_scene_role=SceneRole.SETUP,
        version=1,
    )
    intent = NarrativeIntent(
        intent_id="intent_new",
        scene_role=SceneRole.DISCOVERY,
        objective="测试新线索",
        consequences=[
            NarrativeConsequence(
                target="knowledge.new",
                operation="set",
                value=True
            )
        ]
    )
    original_version = previous.version
    original_threads = previous.unresolved_threads.copy()

    _ = updater.update(previous, intent, [])

    assert previous.version == original_version
    assert previous.unresolved_threads == original_threads


def test_same_input_same_projection():
    """验证 ProjectionUpdater 是确定性的"""
    updater = ProjectionUpdater()
    previous = NarrativeProjection(
        projection_id="test",
        chapter_id="ch1",
        active_conflict="冲突",
        unresolved_threads=["thread_a"],
        last_intent_id="intent_prev",
        last_scene_role=SceneRole.SETUP,
        version=1,
    )
    intent = NarrativeIntent(
        intent_id="intent",
        scene_role=SceneRole.DISCOVERY,
        objective="确定性问题",
        consequences=[]
    )

    p1 = updater.update(previous, intent, [])
    p2 = updater.update(previous, intent, [])

    assert p1.version == p2.version
    assert p1.active_conflict == p2.active_conflict
    assert p1.unresolved_threads == p2.unresolved_threads
    assert p1.last_intent_id == p2.last_intent_id