"""
Phase 13.2: Projection Integration 测试
模拟完整的 Plan-Writer-Validate 流程，验证 Projection 连续更新。
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from src.writing.narrative_projection import NarrativeProjection
from src.writing.narrative_intent import NarrativeIntent, SceneRole, NarrativeConsequence
from src.writing.events import DiscoveryEvent, PlotFlagSetEvent
from src.writing.projection_updater import ProjectionUpdater
from src.writing.projection_service import NarrativeProjectionService
from src.writing.projection_repository import FileProjectionRepository
from src.writing.planner_output import PlannerOutput
from src.writing.planning_contract import PlanningContract


@pytest.fixture
def tmp_repo(tmp_path):
    return FileProjectionRepository(tmp_path)


@pytest.fixture
def service(tmp_repo):
    return NarrativeProjectionService(repository=tmp_repo)


def test_service_save_load(service, tmp_repo):
    """验证 Service 保存和加载"""
    proj = NarrativeProjection(
        projection_id="test",
        chapter_id="chapter_001",
        active_conflict="测试冲突",
        unresolved_threads=["thread_a"],
        last_intent_id="intent",
        last_scene_role=SceneRole.SETUP,
        version=1,
    )
    service.save(proj)
    loaded = service.load_current()
    assert loaded is not None
    assert loaded.projection_id == proj.projection_id
    assert loaded.version == 1


def test_updater_deterministic(service):
    """验证 ProjectionUpdater 确定性，模拟多章节"""
    updater = ProjectionUpdater()
    projection = None

    # 第1章：引入线索
    intent1 = NarrativeIntent(
        intent_id="ch1_intent",
        scene_role=SceneRole.DISCOVERY,
        objective="发现古老玉佩",
        consequences=[
            NarrativeConsequence(
                target="knowledge.ancient_jade",
                operation="set",
                value=True
            )
        ]
    )
    events1 = [
        DiscoveryEvent(
            discoverer="林逸",
            discovery="发现古老玉佩",
            importance="high"
        )
    ]
    projection = updater.update(projection, intent1, events1)
    # 设置 chapter_id 为 chapter_001
    projection = projection.model_copy(update={"chapter_id": "chapter_001"})
    service.save(projection)

    assert len(projection.unresolved_threads) == 1
    assert "ancient_jade" in projection.unresolved_threads[0]
    assert projection.version == 1

    # 第2章：追查线索（未解决）
    intent2 = NarrativeIntent(
        intent_id="ch2_intent",
        scene_role=SceneRole.CONFLICT_ESCALATION,
        objective="追查玉佩来源",
        consequences=[]
    )
    events2 = [
        DiscoveryEvent(
            discoverer="林逸",
            discovery="玉佩与师门有关",
            importance="high"
        )
    ]
    projection = updater.update(projection, intent2, events2)
    projection = projection.model_copy(update={"chapter_id": "chapter_002"})
    service.save(projection)

    # 线程应保留
    assert len(projection.unresolved_threads) >= 1
    assert "ancient_jade" in projection.unresolved_threads[0]
    assert projection.version == 2

    # 第3章：解决线索
    intent3 = NarrativeIntent(
        intent_id="ch3_intent",
        scene_role=SceneRole.DISCOVERY,
        objective="查清玉佩秘密",
        consequences=[]
    )
    events3 = [
        DiscoveryEvent(
            discoverer="林逸",
            discovery="玉佩秘密已查清 ancient_jade",
            importance="critical"
        ),
        PlotFlagSetEvent(
            flag="solved_ancient_jade",
            value=True
        )
    ]
    projection = updater.update(projection, intent3, events3)
    projection = projection.model_copy(update={"chapter_id": "chapter_003"})
    service.save(projection)

    # 线程应被移除
    assert "ancient_jade" not in projection.unresolved_threads
    assert projection.version == 3

    # 验证最新加载
    latest = service.load_current()
    assert latest is not None
    assert latest.version == 3