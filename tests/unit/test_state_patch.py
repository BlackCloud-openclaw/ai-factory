# tests/unit/test_state_patch.py
import pytest
from src.orchestrator.state_patch import StatePatch, WorkflowPhase


def test_state_patch_to_dict_only_non_none():
    patch = StatePatch(
        current_scene_index=5,
        phase=WorkflowPhase.WRITING,
        needs_retry=False,  # False 也会被包含，因为是 bool 且非 None
    )
    d = patch.to_dict()
    assert "current_scene_index" in d
    assert d["current_scene_index"] == 5
    assert "phase" in d
    assert d["phase"] == WorkflowPhase.WRITING
    assert "needs_retry" in d
    assert d["needs_retry"] is False
    assert "current_chapter" not in d  # 未设置，未出现


def test_state_patch_ignores_none_fields():
    patch = StatePatch(current_chapter=2, current_volume=None)
    d = patch.to_dict()
    assert "current_chapter" in d
    assert "current_volume" not in d