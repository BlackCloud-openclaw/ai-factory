import pytest
from unittest.mock import patch, AsyncMock
from src.orchestrator.nodes import validate_node
from src.orchestrator.state import AgentState
from src.orchestrator.state_patch import WorkflowPhase
from src.writing.services.scene_completion import SceneCompletionResult
from src.writing.services.chapter_transition import ChapterTransitionResult


@pytest.mark.asyncio
async def test_validate_node_passes_and_advances_scene():
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

    with patch("src.orchestrator.nodes.ValidatorAgent.run", new_callable=AsyncMock) as mock_validator:
        mock_validator.return_value = {
            "validation_result": {
                "passed": True,
                "parsed_output": {"events": [], "scene_text": "ok"}
            }
        }
        with patch("src.orchestrator.nodes.SceneCompletionService.execute", new_callable=AsyncMock) as mock_scene:
            mock_scene.return_value = SceneCompletionResult(
                state_patch=type('Patch', (), {'current_scene_index': 3, 'phase': WorkflowPhase.WRITING})(),
                chapter_finished=False,
                events_applied=0
            )
            updates = await validate_node(state)

    assert updates.get("current_scene_index") == 3
    assert updates.get("phase") == WorkflowPhase.WRITING


@pytest.mark.asyncio
async def test_validate_node_triggers_chapter_transition_when_finished():
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

    with patch("src.orchestrator.nodes.ValidatorAgent.run", new_callable=AsyncMock) as mock_validator:
        mock_validator.return_value = {
            "validation_result": {
                "passed": True,
                "parsed_output": {"events": [], "scene_text": "ok"}
            }
        }
        with patch("src.orchestrator.nodes.SceneCompletionService.execute", new_callable=AsyncMock) as mock_scene:
            mock_scene.return_value = SceneCompletionResult(
                state_patch=type('Patch', (), {'current_scene_index': 5, 'phase': WorkflowPhase.TRANSITIONING})(),
                chapter_finished=True
            )
        with patch("src.orchestrator.nodes.ChapterTransitionService.execute", new_callable=AsyncMock) as mock_transition:
            mock_transition.return_value = ChapterTransitionResult(
                state_patch=type('Patch', (), {'current_chapter': 2, 'current_scene_index': 0, 'phase': WorkflowPhase.PLANNING})(),
                volume_finished=False
            )
            updates = await validate_node(state)

    assert updates.get("current_chapter") == 2
    assert updates.get("phase") == WorkflowPhase.PLANNING


@pytest.mark.asyncio
async def test_validate_node_retries_on_failure():
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

    with patch("src.orchestrator.nodes.ValidatorAgent.run", new_callable=AsyncMock) as mock_validator:
        mock_validator.return_value = {
            "validation_result": {
                "passed": False,
                "should_retry": True,
                "feedback": "Missing must_events"
            }
        }
        updates = await validate_node(state)

    assert updates.get("retry_count") == 1
    assert updates.get("needs_retry") is True
    assert updates.get("phase") == WorkflowPhase.WRITING


@pytest.mark.asyncio
async def test_validate_node_skips_scene_after_max_retries():
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

    with patch("src.orchestrator.nodes.ValidatorAgent.run", new_callable=AsyncMock) as mock_validator:
        mock_validator.return_value = {
            "validation_result": {
                "passed": False,
                "should_retry": True,
                "feedback": "Still missing"
            }
        }
        with patch("src.orchestrator.nodes._skip_scene", new_callable=AsyncMock):
            updates = await validate_node(state)

    assert updates.get("current_scene_index") == 3
    assert updates.get("retry_count") == 0