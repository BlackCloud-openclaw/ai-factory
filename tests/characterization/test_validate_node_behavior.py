"""
Characterization Tests for validate_node

这些测试捕获 validate_node 的核心行为，防止重构时引入语义变化。
"""

import pytest
from unittest.mock import patch
from src.orchestrator.nodes import validate_node
from src.orchestrator.state import AgentState
from src.orchestrator.state_patch import StatePatch, WorkflowPhase
from src.writing.services.scene_completion import SceneCompletionResult
from src.writing.services.chapter_transition import ChapterTransitionResult


@pytest.mark.asyncio
async def test_validate_node_passes_and_advances_scene():
    """验证通过时，场景索引增加，阶段为 WRITING"""
    state = AgentState(
        novel_id="test",
        current_volume=1,
        current_chapter=1,
        current_scene_index=2,
        total_scenes_in_chapter=5,
        validation_mode="novel",
        scene_text="some text",
        scene_plan={},
        current_state={},
        retry_count=0,
        max_retries_per_subtask=3,
    )

    async def fake_validator_run(_):
        return {
            "validation_result": {
                "passed": True,
                "parsed_output": {"events": [], "scene_text": "ok"}
            }
        }

    # 创建真正的 StatePatch 对象
    patch_obj = StatePatch(
        current_scene_index=3,
        phase=WorkflowPhase.WRITING,
        current_state={}
    )

    async def fake_scene_completion_execute(_):
        return SceneCompletionResult(
            state_patch=patch_obj,
            chapter_finished=False,
            events_applied=0
        )

    with patch("src.orchestrator.nodes.ValidatorAgent.run", side_effect=fake_validator_run):
        with patch("src.orchestrator.nodes.SceneCompletionService.execute", side_effect=fake_scene_completion_execute):
            updates = await validate_node(state)

    assert updates.get("current_scene_index") == 3
    assert updates.get("phase") == WorkflowPhase.WRITING


@pytest.mark.asyncio
async def test_validate_node_triggers_chapter_transition_when_finished():
    """章节完成时，触发切换，阶段变为 PLANNING"""
    state = AgentState(
        novel_id="test",
        current_volume=1,
        current_chapter=1,
        current_scene_index=4,
        total_scenes_in_chapter=5,
        validation_mode="novel",
        scene_text="some text",
        scene_plan={},
        current_state={},
        total_chapters_in_volume=10,
        outline={"volumes": [{"chapters": [{}]*10}]},
    )

    async def fake_validator_run(_):
        return {
            "validation_result": {
                "passed": True,
                "parsed_output": {"events": [], "scene_text": "ok"}
            }
        }

    scene_patch = StatePatch(
        current_scene_index=5,
        phase=WorkflowPhase.TRANSITIONING,
        current_state={}
    )
    transition_patch = StatePatch(
        current_chapter=2,
        current_scene_index=0,
        phase=WorkflowPhase.PLANNING,
        current_state={}
    )

    async def fake_scene_completion_execute(_):
        return SceneCompletionResult(
            state_patch=scene_patch,
            chapter_finished=True
        )

    async def fake_chapter_transition_execute(_):
        return ChapterTransitionResult(
            state_patch=transition_patch,
            volume_finished=False
        )

    with patch("src.orchestrator.nodes.ValidatorAgent.run", side_effect=fake_validator_run):
        with patch("src.orchestrator.nodes.SceneCompletionService.execute", side_effect=fake_scene_completion_execute):
            with patch("src.orchestrator.nodes.ChapterTransitionService.execute", side_effect=fake_chapter_transition_execute):
                updates = await validate_node(state)

    assert updates.get("current_chapter") == 2
    assert updates.get("phase") == WorkflowPhase.PLANNING


@pytest.mark.asyncio
async def test_validate_node_retries_on_failure():
    """验证失败且可重试时，重试计数增加，阶段保持 WRITING"""
    state = AgentState(
        novel_id="test",
        current_scene_index=2,
        validation_mode="novel",
        retry_count=0,
        max_retries_per_subtask=3,
        scene_text="bad",
        scene_plan={},
        current_state={}
    )

    async def fake_validator_run(_):
        return {
            "validation_result": {
                "passed": False,
                "should_retry": True,
                "feedback": "Missing must_events"
            }
        }

    with patch("src.orchestrator.nodes.ValidatorAgent.run", side_effect=fake_validator_run):
        updates = await validate_node(state)

    assert updates.get("retry_count") == 1
    assert updates.get("needs_retry") is True
    assert updates.get("phase") == WorkflowPhase.WRITING


@pytest.mark.asyncio
async def test_validate_node_skips_scene_after_max_retries():
    """超过最大重试次数时，跳过当前场景，索引+1"""
    state = AgentState(
        novel_id="test",
        current_scene_index=2,
        validation_mode="novel",
        retry_count=3,
        max_retries_per_subtask=3,
        scene_text="bad",
        scene_plan={},
        current_state={}
    )

    async def fake_validator_run(_):
        return {
            "validation_result": {
                "passed": False,
                "should_retry": True,
                "feedback": "Still missing"
            }
        }

    async def fake_skip_scene(_):
        pass

    with patch("src.orchestrator.nodes.ValidatorAgent.run", side_effect=fake_validator_run):
        with patch("src.orchestrator.nodes._skip_scene", side_effect=fake_skip_scene):
            updates = await validate_node(state)

    assert updates.get("current_scene_index") == 3
    assert updates.get("retry_count") == 0