# tests/characterization/test_validate_node_behavior.py
"""
Characterization tests for validate_node.
这些测试锁定当前行为，重构后应继续通过。
"""
import pytest
from unittest.mock import AsyncMock, patch
from src.orchestrator.nodes import validate_node
from src.orchestrator.state import AgentState
from src.orchestrator.state_patch import WorkflowPhase


@pytest.mark.asyncio
async def test_validate_node_passes_and_advances_scene():
    """当验证通过时，应该推进场景索引，phase 变为 WRITING 或 TRANSITIONING"""
    state = AgentState(
        novel_id="test_novel",
        current_volume=1,
        current_chapter=1,
        current_scene_index=0,
        total_scenes_in_chapter=3,
        current_state={"characters": {}},
        scene_plan={"goal": "test goal"},
        scene_text="Some scene text",
        task_type="scene_plan",
    )
    
    # Mock ValidatorAgent 返回成功
    with patch("src.orchestrator.nodes.ValidatorAgent") as MockValidator:
        mock_validator = AsyncMock()
        mock_validator.run.return_value = {
            "validation_result": {
                "passed": True,
                "should_retry": False,
                "parsed_output": {
                    "events": [],
                    "scene_text": "validated scene"
                }
            }
        }
        MockValidator.return_value = mock_validator
        
        # Mock SceneCompletionService 及其依赖（避免真实 DB）
        with patch("src.orchestrator.nodes.SceneCompletionService") as MockService:
            mock_service = AsyncMock()
            mock_service.execute.return_value = SceneCompletionResult(
                state_patch=StatePatch(
                    current_scene_index=1,
                    phase=WorkflowPhase.WRITING
                ),
                chapter_finished=False
            )
            MockService.return_value = mock_service
            
            result_dict = await validate_node(state)
    
    # 验证返回的 patch 包含了预期的状态变更
    assert result_dict.get("current_scene_index") == 1
    # 注意：实际返回的是 dict，包含 phase 吗？取决于 service 返回的 patch
    # 这里只做示例，实际需根据业务调整


@pytest.mark.asyncio
async def test_validate_node_retries_on_failure():
    """当验证失败且应该重试时，返回需要重试的 patch"""
    state = AgentState(
        novel_id="test_novel",
        current_volume=1,
        current_chapter=1,
        current_scene_index=0,
        total_scenes_in_chapter=3,
        retry_count=0,
        max_retries_per_subtask=2,
        # ... 其他必要字段
    )
    
    with patch("src.orchestrator.nodes.ValidatorAgent") as MockValidator:
        mock_validator = AsyncMock()
        mock_validator.run.return_value = {
            "validation_result": {
                "passed": False,
                "should_retry": True,
                "feedback": "Missing must_event"
            }
        }
        MockValidator.return_value = mock_validator
        
        result_dict = await validate_node(state)
    
    assert result_dict.get("needs_retry") is True
    assert result_dict.get("retry_count") == 1
    assert result_dict.get("writing_feedback") == "Missing must_event"