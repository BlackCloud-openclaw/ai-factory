# tests/phase14/test_validator_agent_adapter.py

import pytest
from unittest.mock import AsyncMock, patch
from src.agents.validator import ValidatorAgent
from src.orchestrator.state import AgentState
from src.writing.validation_result import ValidatorOutput, ValidationStage, Violation, ViolationSeverity


@pytest.mark.asyncio
async def test_validator_returns_validator_output():
    """测试 run 方法返回包含 validator_output 的字典"""
    state = AgentState(
        project_id="test_project",
        metadata={"execution_id": "exec_test_123"},
        scene_text='{"scene_text": "测试文本", "events": []}',
        validation_mode="novel",
    )

    agent = ValidatorAgent()
    
    with patch.object(agent, "_validate_novel_enhanced") as mock_validate:
        # 构造完整的返回字典
        output = ValidatorOutput.success("exec_test_123")
        mock_validate.return_value = {
            "passed": True,
            "feedback": "校验通过",
            "suggestions": [],
            "should_retry": False,
            "error_details": {},
            "parsed_output": {"scene_text": "测试文本", "events": []},
            "loop_advancement_score": 0.0,
            "control_scores": {},
            "validator_output": output.to_runtime_dict(),
        }
        
        result = await agent.run(state)
        
        # 检查返回结构
        assert "validator_output" in result
        assert "validation_result" in result
        assert "final_answer" in result
        
        output_data = result["validator_output"]
        assert output_data["execution_id"] == "exec_test_123"
        assert output_data["status"] == "passed"
        assert result["validation_result"]["passed"] is True


@pytest.mark.asyncio
async def test_validator_output_with_errors():
    """测试错误场景下返回的 validator_output 包含错误信息"""
    state = AgentState(
        project_id="test_project",
        metadata={"execution_id": "exec_test_456"},
        scene_text='{"scene_text": "无效文本", "events": []}',
        validation_mode="novel",
    )

    agent = ValidatorAgent()
    
    with patch.object(agent, "_validate_novel_enhanced") as mock_validate:
        # 构造包含错误的返回
        error_violation = Violation(
            rule_id="TEST_ERROR",
            severity=ViolationSeverity.ERROR,
            description="验证失败"
        )
        output = ValidatorOutput.failure("exec_test_456", [error_violation])
        mock_validate.return_value = {
            "passed": False,
            "feedback": "验证失败",
            "suggestions": ["修复错误"],
            "should_retry": True,
            "error_details": {"missing_events": ["事件A"]},
            "parsed_output": {},
            "loop_advancement_score": 0.0,
            "control_scores": {},
            "validator_output": output.to_runtime_dict(),
        }
        
        result = await agent.run(state)
        
        # 检查错误状态
        assert "validator_output" in result
        output_data = result["validator_output"]
        assert output_data["status"] == "failed"
        assert len(output_data["violations"]) == 1
        assert output_data["violations"][0]["rule_id"] == "TEST_ERROR"
        # 验证顶层 validation_result
        assert result["validation_result"]["passed"] is False


@pytest.mark.asyncio
async def test_validator_generates_execution_id():
    """测试当 state 没有 execution_id 时自动生成"""
    state = AgentState(
        project_id="test_project",
        metadata={},  # 无 execution_id
        scene_text='{"scene_text": "测试文本", "events": []}',
        validation_mode="novel",
    )

    agent = ValidatorAgent()
    
    # 我们需要 mock 以便让 _validate_novel_enhanced 返回一个包含 validator_output 的字典
    # 但我们也希望实际的 execution_id 由 run 方法生成
    async def mock_validate(*args, **kwargs):
        # 获取传入的 execution_id
        passed_execution_id = kwargs.get("execution_id", "")
        # 构造 ValidatorOutput，使用传入的 execution_id
        output = ValidatorOutput.success(passed_execution_id)
        return {
            "passed": True,
            "feedback": "校验通过",
            "suggestions": [],
            "should_retry": False,
            "error_details": {},
            "parsed_output": {"scene_text": "测试文本", "events": []},
            "loop_advancement_score": 0.0,
            "control_scores": {},
            "validator_output": output.to_runtime_dict(),
        }
    
    with patch.object(agent, "_validate_novel_enhanced", new=mock_validate):
        result = await agent.run(state)
        
        assert "validator_output" in result
        output_data = result["validator_output"]
        assert "execution_id" in output_data
        # 验证生成的 execution_id 格式
        assert output_data["execution_id"].startswith("val_test_project_")