import pytest
from src.writing.services.scene_planning import ScenePlanningService
from src.writing.services.models import ScenePlanningCommand, ScenePlanningResult
from src.orchestrator.state_patch import StatePatch
from src.agents.planner import PlannerAgent
from src.agents.drama_planner import DramaPlannerAgent
from src.orchestrator.state import AgentState
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_scene_planning_result_contract_field_exists():
    """验证 ScenePlanningResult dataclass 包含 planner_outputs 字段"""
    result = ScenePlanningResult(
        state_patch=StatePatch(),
        total_scenes=3,
        planner_outputs=[{"narrative_intent": {}, "execution_contract": {}}],
    )
    assert hasattr(result, "planner_outputs")
    assert isinstance(result.planner_outputs, list)
    assert len(result.planner_outputs) == 1


@pytest.mark.asyncio
async def test_scene_planning_result_default_empty_list():
    """验证 planner_outputs 默认值为空列表"""
    result = ScenePlanningResult(
        state_patch=StatePatch(metadata={"some_key": "value"}),
        total_scenes=0,
    )
    assert result.planner_outputs == []


@pytest.mark.asyncio
async def test_scene_planning_service_returns_planner_outputs():
    """验证 ScenePlanningService 正确返回 planner_outputs。"""
    cmd = ScenePlanningCommand(
        novel_id="test_novel",
        volume=1,
        chapter=1,
        task_type="scene_plan",
        outline={},
        current_state={},
        user_input="test",
        resume=False,
        total_chapters_in_volume=10,
        metadata={},
        intent_resolver=None,
    )

    # 模拟 PlannerAgent.run 返回预设数据
    mock_planner_output = {
        "narrative_intent": {
            "intent_id": "test_intent",
            "scene_role": "setup",
            "objective": "测试目标",
            "preconditions": [],
            "beats": [],
            "consequences": [],
            "interaction_plan": None,
        },
        "execution_contract": {
            "version": "1.0",
            "scene_id": "scene_1",
            "intent": {"goal": "test", "conflict": "test", "expected_outcome": "test"},
            "execution": {"units": []},
            "observables": {"state_changes": [], "story_events": [], "narrative_flags": []},
            "constraints": [],
            "metadata": {"chapter": 1, "scene_index": 0},
        }
    }

    with patch("src.writing.services.scene_planning.PlannerAgent") as MockPlanner:
        mock_planner = AsyncMock()
        mock_planner.run.return_value = {
            "scene_plan": {"scenes": [{"goal": "test"}]},
            "planner_outputs": [mock_planner_output],
        }
        MockPlanner.return_value = mock_planner

        result = await ScenePlanningService.execute(cmd)
        assert len(result.planner_outputs) == 1
        assert result.planner_outputs[0]["narrative_intent"]["objective"] == "测试目标" 


@pytest.mark.asyncio
async def test_scene_planning_service_handles_empty_planner_outputs():
    """
    验证当 PlannerAgent 无 planner_outputs 时，Service 返回空列表
    """
    async def mock_drama_run(self, state):
        return {"drama_structure": {"scene_role": "ESCALATION"}}

    async def mock_planner_run(self, state):
        # 不设置 planner_outputs
        return {"scene_plan": {"scenes": [{"goal": "test"}]}}

    with patch.object(PlannerAgent, 'run', new=mock_planner_run), \
         patch.object(DramaPlannerAgent, 'run', new=mock_drama_run):
        cmd = ScenePlanningCommand(
            novel_id="test_novel",
            volume=1,
            chapter=1,
            task_type="scene_plan",
            outline={"volumes": [{"chapters": [{}]}]},
            current_state={},
            user_input="test",
            metadata={},
        )

        result = await ScenePlanningService.execute(cmd)

        assert result.planner_outputs == []
        assert result.state_patch is not None
        assert "planner_outputs" not in cmd.metadata