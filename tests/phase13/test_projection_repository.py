"""
Phase 13.2: ProjectionRepository 测试
"""

import pytest
from pathlib import Path
from src.writing.narrative_projection import NarrativeProjection
from src.writing.narrative_intent import SceneRole
from src.writing.projection_repository import FileProjectionRepository


def test_file_repository_save_load(tmp_path):
    """测试保存和加载"""
    repo = FileProjectionRepository(tmp_path)
    proj = NarrativeProjection(
        projection_id="test_proj",
        chapter_id="chapter_001",
        active_conflict="测试冲突",
        last_intent_id="intent",
        last_scene_role=SceneRole.SETUP,
        version=1,
    )
    repo.save(proj)
    loaded = repo.load("chapter_001")
    assert loaded is not None
    assert loaded.projection_id == proj.projection_id
    assert loaded.chapter_id == proj.chapter_id
    assert loaded.active_conflict == "测试冲突"
    assert loaded.version == 1


def test_file_repository_latest(tmp_path):
    """测试最新投影加载"""
    repo = FileProjectionRepository(tmp_path)
    proj1 = NarrativeProjection(
        projection_id="proj1",
        chapter_id="chapter_001",
        active_conflict="冲突1",
        last_intent_id="intent1",
        last_scene_role=SceneRole.SETUP,
        version=1,
    )
    proj2 = NarrativeProjection(
        projection_id="proj2",
        chapter_id="chapter_002",
        active_conflict="冲突2",
        last_intent_id="intent2",
        last_scene_role=SceneRole.CONFLICT_ESCALATION,
        version=1,
    )
    repo.save(proj1)
    repo.save(proj2)
    latest = repo.latest()
    assert latest is not None
    assert latest.chapter_id == "chapter_002"
    assert latest.active_conflict == "冲突2"


def test_file_repository_load_missing(tmp_path):
    """测试加载不存在的投影"""
    repo = FileProjectionRepository(tmp_path)
    assert repo.load("chapter_999") is None
    assert repo.latest() is None