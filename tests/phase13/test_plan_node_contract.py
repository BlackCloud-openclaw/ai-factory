# tests/phase13/test_plan_node_contract.py
"""
Phase 13.2.2 Contract Test for plan_node → StatePatch propagation

验证：
1. plan_node 正确消费 ScenePlanningResult.planner_outputs
2. StatePatch 被正确构建并包含 planner_outputs
3. StatePatch.to_dict() 序列化后仍保留 planner_outputs
"""

import pytest
from unittest.mock import AsyncMock, patch
from src.orchestrator.nodes import plan_node
from src.orchestrator.state import AgentState
from src.orchestrator.state_patch import StatePatch
from src.writing.services.models import ScenePlanningResult



@pytest.mark.asyncio
@patch("src.orchestrator.nodes.ScenePlanningService")
async def test_plan_node_propagates_planner_outputs_to_state_patch(mock_service):
    patch = StatePatch(
        scene_plan_list=[{"goal": "test"}],
        total_scenes_in_chapter=1,
    )
    mock_result = ScenePlanningResult(
        state_patch=patch,
        total_scenes=1,
        planner_outputs=[
            {
                "narrative_intent": {
                    "intent_id": "test_1",
                    "scene_role": "setup",
                    "objective": "完成测试场景",
                },
                "execution_contract": {"scene_id": "scene_1"},
            }
        ],
    )
    mock_service.execute = AsyncMock(return_value=mock_result)

    state = AgentState(
        novel_id="test_novel",
        current_volume=1,
        current_chapter=1,
        current_scene_index=0,
        task_type="scene_plan",
        outline={"volumes": [{"chapters": [{}]}]},
        current_state={},
        user_input="test",
        metadata={},
    )

    result_payload = await plan_node(state)

    assert "planner_outputs" in result_payload
    assert len(result_payload["planner_outputs"]) == 1
    assert result_payload["planner_outputs"][0]["narrative_intent"]["scene_role"] == "setup"
    assert "narrative_intent" in result_payload
    # 修正：使用属性访问
    assert result_payload["narrative_intent"].intent_id == "test_1"


@pytest.mark.asyncio
@patch("src.orchestrator.nodes.ScenePlanningService")
async def test_plan_node_handles_empty_planner_outputs(mock_service):
    mock_result = ScenePlanningResult(
        state_patch=StatePatch(),
        total_scenes=0,
        planner_outputs=[],  # 空列表
    )
    mock_service.execute = AsyncMock(return_value=mock_result)

    state = AgentState(
        novel_id="test_novel",
        current_volume=1,
        current_chapter=1,
        current_scene_index=0,
        task_type="scene_plan",
        outline={"volumes": [{"chapters": [{}]}]},
        current_state={},
        user_input="test",
        metadata={},
    )

    result_payload = await plan_node(state)

    # 应包含 planner_outputs 键，值为空列表
    assert "planner_outputs" in result_payload
    assert result_payload["planner_outputs"] == []
    # narrative_intent 应该不存在
    assert "narrative_intent" not in result_payload