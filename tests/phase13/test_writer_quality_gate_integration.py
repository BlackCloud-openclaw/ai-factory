# tests/phase13/test_writer_quality_gate_integration.py
"""
ControlledWriter + QualityGate 集成测试
验证 feedback 是否真正进入 prompt 闭环。
"""

import pytest
from unittest.mock import MagicMock, patch

from src.writing.controlled_writer import ControlledWriter
from src.writing.quality_gate import QualityGate
from src.writing.validation import SemanticValidator, ValidationResult, ValidationEvidence, SignalSource
from src.writing.planning_contract import PlanningContract, Intent, Execution, ExecutionUnit, Observables, ContractMetadata


def create_mock_contract():
    """创建模拟的 WritingContract，包含 execution_contract 属性。"""
    planning_contract = PlanningContract(
        version="1.0",
        scene_id="test_scene",
        intent=Intent(goal="测试目标", conflict="测试冲突", expected_outcome="测试结果"),
        execution=Execution(units=[
            ExecutionUnit(id="U1", label="action", description="获得玉佩")
        ]),
        observables=Observables(state_changes=[]),
        constraints=[],
        metadata=ContractMetadata(chapter=1, scene_index=0),
    )
    # 使用 MagicMock 模拟 WritingContract
    mock_contract = MagicMock()
    mock_contract.execution_contract = planning_contract
    # 如果其他属性被使用，可以添加
    return mock_contract


@pytest.mark.asyncio
async def test_writer_retry_feedback_injected_into_prompt():
    """验证 QualityGate retry 时 feedback 进入下一轮 prompt"""
    # 1. Mock SemanticValidator
    validator_mock = MagicMock(spec=SemanticValidator)

    fail_result = ValidationResult(
        passed=False,
        missing=["获得玉佩"],
        matched=[],
        blocking_missing=["获得玉佩"],
        overall_confidence=0.0,
        weight_applied=0.0,
        errors=["Blocking missing: 获得玉佩"],
    )
    pass_result = ValidationResult(
        passed=True,
        missing=[],
        matched=[ValidationEvidence(
            evidence_id="e1",
            event_id="evt1",
            event_text="获得玉佩",
            matcher="exact",
            confidence=1.0,
            source=SignalSource.LLM,
            matched_text="获得玉佩",
            weight=1.0,
        )],
        blocking_missing=[],
        overall_confidence=1.0,
        weight_applied=1.0,
    )
    validator_mock.validate.side_effect = [fail_result, pass_result]

    # 2. 创建 Writer
    writer = ControlledWriter(
        max_retries_per_segment=2,
        semantic_validator=validator_mock,
        quality_gate=QualityGate(),
    )

    contract = create_mock_contract()
    units = contract.execution_contract.execution.units if contract.execution_contract else []

    # 记录 prompt 构建调用
    prompt_calls = []

    async def mock_call_llm(prompt, max_tokens=None):
        if "获得玉佩" in prompt:
            return '{"scene_text": "林逸获得玉佩", "events": [{"type": "plot_flag_set", "flag": "获得玉佩", "value": true}]}', {}
        else:
            return '{"scene_text": "没有玉佩的文本", "events": []}', {}

    def mock_parse(text):
        if "获得玉佩" in text:
            return MagicMock(scene_text="林逸获得玉佩", events=[{"type": "plot_flag_set", "flag": "获得玉佩", "value": True}])
        else:
            return MagicMock(scene_text="没有玉佩的文本", events=[])

    def mock_build_prompt(**kwargs):
        prompt_calls.append(kwargs)
        error_hint = kwargs.get('error_hint', '')
        if "获得玉佩" in error_hint:
            return f"请生成包含获得玉佩的场景。提示：{error_hint}"
        return "普通 prompt"

    with patch.object(writer, '_call_llm', side_effect=mock_call_llm):
        with patch.object(writer, '_parse_and_validate', side_effect=mock_parse):
            with patch.object(writer, '_build_segment_prompt', side_effect=mock_build_prompt):
                writer.enable_fallback = False

                result_text, result_events, success = await writer._execute_segment(
                    contract=contract,
                    units=units,
                    idx=0,
                    total=1,
                    previous_text="",
                    previous_events=[],
                    current_state={}
                )

                assert success is True
                assert "获得玉佩" in result_text
                assert len(prompt_calls) >= 2

                # 关键验证：第二次 prompt 调用包含第一次的 feedback
                second_call = prompt_calls[1] if len(prompt_calls) > 1 else prompt_calls[-1]
                error_hint = second_call.get('error_hint', '')
                assert "获得玉佩" in error_hint or "关键事件" in error_hint

                assert validator_mock.validate.call_count == 2


@pytest.mark.asyncio
async def test_writer_passes_on_first_success():
    """验证第一次成功时直接通过，不触发 retry"""
    validator_mock = MagicMock(spec=SemanticValidator)
    pass_result = ValidationResult(
        passed=True,
        missing=[],
        matched=[ValidationEvidence(
            evidence_id="e1",
            event_id="evt1",
            event_text="获得玉佩",
            matcher="exact",
            confidence=1.0,
            source=SignalSource.LLM,
            matched_text="获得玉佩",
            weight=1.0,
        )],
        blocking_missing=[],
        overall_confidence=1.0,
        weight_applied=1.0,
    )
    validator_mock.validate.return_value = pass_result

    writer = ControlledWriter(
        max_retries_per_segment=2,
        semantic_validator=validator_mock,
        quality_gate=QualityGate(),
    )

    contract = create_mock_contract()
    units = contract.execution_contract.execution.units if contract.execution_contract else []

    async def mock_call_llm(prompt, max_tokens=None):
        return '{"scene_text": "林逸获得玉佩", "events": [{"type": "plot_flag_set", "flag": "获得玉佩", "value": true}]}', {}

    def mock_parse(text):
        return MagicMock(scene_text="林逸获得玉佩", events=[{"type": "plot_flag_set", "flag": "获得玉佩", "value": True}])

    prompt_calls = []

    def mock_build_prompt(**kwargs):
        prompt_calls.append(kwargs)
        return "prompt"

    with patch.object(writer, '_call_llm', side_effect=mock_call_llm):
        with patch.object(writer, '_parse_and_validate', side_effect=mock_parse):
            with patch.object(writer, '_build_segment_prompt', side_effect=mock_build_prompt):
                result_text, result_events, success = await writer._execute_segment(
                    contract=contract,
                    units=units,
                    idx=0,
                    total=1,
                    previous_text="",
                    previous_events=[],
                    current_state={}
                )

                assert success is True
                assert len(prompt_calls) == 1
                assert prompt_calls[0].get('error_hint', '') == ''
                assert validator_mock.validate.call_count == 1