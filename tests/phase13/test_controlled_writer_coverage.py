# tests/phase13/test_controlled_writer_coverage.py
"""
Phase 13.2.3E: ControlledWriter Coverage Tests
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any
import json

from src.writing.controlled_writer import ControlledWriter
from src.writing.quality_gate import QualityGate, QualityGateResult
from src.writing.validation import SemanticValidator, ValidationResult, ValidationEvidence, SignalSource, NoOpEmbeddingProvider
from src.writing.contracts import WritingContract
from src.writing.planning_contract import (
    PlanningContract, Intent, Execution, ExecutionUnit, Observables, ContractMetadata, StateChange
)
from src.writing.scene_execution_context import SceneExecutionContext


# ============================================================================
# Test Helpers
# ============================================================================

def create_contract_with_units(num_units: int) -> WritingContract:
    planning = PlanningContract(
        version="1.0",
        scene_id="test_scene",
        intent=Intent(goal="测试目标", conflict="测试冲突", expected_outcome="测试结果"),
        execution=Execution(units=[
            ExecutionUnit(id=f"U{i}", label="action", description=f"事件 {i}")
            for i in range(1, num_units + 1)
        ]),
        observables=Observables(state_changes=[]),
        constraints=[],
        metadata=ContractMetadata(chapter=1, scene_index=0),
    )
    mock = MagicMock(spec=WritingContract)
    mock.execution_contract = planning
    mock.scene_context = SceneExecutionContext(
        chapter_id="test_chap",
        scene_id="test_scene",
        scene_role="setup",
        dramatic_function="transition",
        characters=["林逸"],
        location="测试地点",
        time="黄昏",
    )
    return mock


def create_validation_result(passed: bool, missing: List[str] = None, blocking: List[str] = None) -> ValidationResult:
    if missing is None:
        missing = []
    if blocking is None:
        blocking = []
    matched = []
    if passed:
        matched = [ValidationEvidence(
            evidence_id="e1",
            event_id="evt1",
            event_text="事件 1",
            matcher="exact",
            confidence=1.0,
            source=SignalSource.LLM,
            matched_text="事件 1",
            weight=1.0,
        )]
    return ValidationResult(
        passed=passed,
        missing=missing,
        matched=matched,
        blocking_missing=blocking,
        overall_confidence=1.0 if passed else 0.0,
        weight_applied=1.0 if passed else 0.0,
        errors=["Blocking missing"] if blocking else [],
    )


# ============================================================================
# Test Cases
# ============================================================================

@pytest.mark.asyncio
async def test_activation_with_1_unit():
    writer = ControlledWriter(max_retries_per_segment=2)
    contract = create_contract_with_units(1)
    units = contract.execution_contract.execution.units
    segments = writer._determine_segments(units)
    assert segments == 1


@pytest.mark.asyncio
async def test_activation_with_3_units():
    writer = ControlledWriter(max_retries_per_segment=2)
    contract = create_contract_with_units(3)
    units = contract.execution_contract.execution.units
    segments = writer._determine_segments(units)
    assert segments == 1


@pytest.mark.asyncio
async def test_activation_with_6_units():
    writer = ControlledWriter(max_retries_per_segment=2)
    contract = create_contract_with_units(6)
    units = contract.execution_contract.execution.units
    segments = writer._determine_segments(units)
    assert segments == 2


@pytest.mark.asyncio
async def test_segmentation_counts():
    writer = ControlledWriter(max_retries_per_segment=2)
    units_1 = [MagicMock() for _ in range(1)]
    units_2 = [MagicMock() for _ in range(2)]
    units_3 = [MagicMock() for _ in range(3)]
    units_4 = [MagicMock() for _ in range(4)]
    units_5 = [MagicMock() for _ in range(5)]
    units_6 = [MagicMock() for _ in range(6)]
    units_7 = [MagicMock() for _ in range(7)]
    units_8 = [MagicMock() for _ in range(8)]
    units_9 = [MagicMock() for _ in range(9)]
    assert writer._determine_segments(units_1) == 1
    assert writer._determine_segments(units_2) == 1
    assert writer._determine_segments(units_3) == 1
    assert writer._determine_segments(units_4) == 1
    assert writer._determine_segments(units_5) == 2
    assert writer._determine_segments(units_6) == 2
    assert writer._determine_segments(units_7) == 2
    assert writer._determine_segments(units_8) == 2
    assert writer._determine_segments(units_9) == 3


@pytest.mark.asyncio
async def test_retry_flow_first_fail_then_pass():
    validator_mock = MagicMock(spec=SemanticValidator)
    validator_mock.validate.side_effect = [
        create_validation_result(False, missing=["事件 1"], blocking=["事件 1"]),
        create_validation_result(True),
    ]
    writer = ControlledWriter(
        max_retries_per_segment=2,
        semantic_validator=validator_mock,
        quality_gate=QualityGate(),
    )
    contract = create_contract_with_units(1)
    units = contract.execution_contract.execution.units

    prompt_calls: List[Dict[str, Any]] = []
    def mock_build_prompt(**kwargs):
        prompt_calls.append(kwargs)
        error_hint = kwargs.get('error_hint', '')
        if error_hint:
            return f"带反馈的提示: {error_hint}"
        return "基础提示"

    llm_responses = [
        '{"scene_text": "缺少关键事件", "events": []}',
        '{"scene_text": "林逸获得玉佩", "events": [{"type": "plot_flag_set", "flag": "获得玉佩", "value": true}]}',
    ]
    response_index = 0
    async def mock_call_llm(prompt, max_tokens=None):
        nonlocal response_index
        if response_index < len(llm_responses):
            resp = llm_responses[response_index]
            response_index += 1
            return resp, {}
        return '{"scene_text": "fallback", "events": []}', {}

    def mock_parse(text):
        return MagicMock(scene_text=text, events=[])

    with patch.object(writer, '_call_llm', side_effect=mock_call_llm):
        with patch.object(writer, '_parse_and_validate', side_effect=mock_parse):
            with patch.object(writer, '_build_segment_prompt', side_effect=mock_build_prompt):
                writer.enable_fallback = False
                text, events, ok = await writer._execute_segment(
                    contract=contract, units=units, idx=0, total=1,
                    previous_text="", previous_events=[], current_state={}
                )

    assert ok is True
    assert "获得玉佩" in text
    assert len(prompt_calls) >= 2
    error_hints = [p.get('error_hint', '') for p in prompt_calls]
    assert any("事件 1" in h for h in error_hints)
    assert validator_mock.validate.call_count == 2


@pytest.mark.asyncio
async def test_retry_feedback_injected_into_prompt():
    validator_mock = MagicMock(spec=SemanticValidator)
    validator_mock.validate.side_effect = [
        create_validation_result(False, missing=["获得玉佩"], blocking=["获得玉佩"]),
        create_validation_result(True),
    ]
    writer = ControlledWriter(
        max_retries_per_segment=2,
        semantic_validator=validator_mock,
        quality_gate=QualityGate(),
    )
    contract = create_contract_with_units(1)
    units = contract.execution_contract.execution.units

    prompt_calls: List[Dict[str, Any]] = []
    def mock_build_prompt(**kwargs):
        prompt_calls.append(kwargs)
        return f"Prompt with error_hint={kwargs.get('error_hint', '')}"

    async def mock_call_llm(prompt, max_tokens=None):
        if "error_hint" in prompt and "获得玉佩" in prompt:
            return '{"scene_text": "林逸获得玉佩", "events": []}', {}
        return '{"scene_text": "没有", "events": []}', {}

    def mock_parse(text):
        return MagicMock(scene_text=text, events=[])

    with patch.object(writer, '_call_llm', side_effect=mock_call_llm):
        with patch.object(writer, '_parse_and_validate', side_effect=mock_parse):
            with patch.object(writer, '_build_segment_prompt', side_effect=mock_build_prompt):
                writer.enable_fallback = False
                text, events, ok = await writer._execute_segment(
                    contract=contract, units=units, idx=0, total=1,
                    previous_text="", previous_events=[], current_state={}
                )

    assert ok is True
    error_hints = [p.get('error_hint', '') for p in prompt_calls]
    assert any("获得玉佩" in h for h in error_hints)


@pytest.mark.asyncio
async def test_force_pass_when_retries_exhausted():
    validator_mock = MagicMock(spec=SemanticValidator)
    validator_mock.validate.return_value = create_validation_result(
        False, missing=["事件 1"], blocking=["事件 1"]
    )
    writer = ControlledWriter(
        max_retries_per_segment=1,
        semantic_validator=validator_mock,
        quality_gate=QualityGate(),
    )
    contract = create_contract_with_units(1)
    units = contract.execution_contract.execution.units

    async def mock_call_llm(prompt, max_tokens=None):
        return '{"scene_text": "失败文本", "events": []}', {}

    def mock_parse(text):
        return MagicMock(scene_text=text, events=[])

    prompt_calls = []
    def mock_build_prompt(**kwargs):
        prompt_calls.append(kwargs)
        return "prompt"

    with patch.object(writer, '_call_llm', side_effect=mock_call_llm):
        with patch.object(writer, '_parse_and_validate', side_effect=mock_parse):
            with patch.object(writer, '_build_segment_prompt', side_effect=mock_build_prompt):
                writer.enable_fallback = False
                text, events, ok = await writer._execute_segment(
                    contract=contract, units=units, idx=0, total=1,
                    previous_text="", previous_events=[], current_state={}
                )

    assert ok is True
    try:
        parsed = json.loads(text)
        scene_text = parsed.get("scene_text", "")
    except:
        scene_text = text
    assert scene_text == "失败文本"
    assert len(prompt_calls) == 2


@pytest.mark.asyncio
async def test_fallback_flow():
    """
    Test fallback path: ensures fallback succeeds when text length > 300.
    """
    validator_mock = MagicMock(spec=SemanticValidator)
    validator_mock.validate.return_value = create_validation_result(
        False, missing=["事件 1"], blocking=["事件 1"]
    )

    quality_gate_mock = MagicMock(spec=QualityGate)
    quality_gate_mock.evaluate.return_value = QualityGateResult(
        decision="retry",
        score=0.0,
        feedback="缺失关键事件",
        details={}
    )

    writer = ControlledWriter(
        max_retries_per_segment=2,
        semantic_validator=validator_mock,
        quality_gate=quality_gate_mock,
        enable_fallback=True,
    )

    contract = create_contract_with_units(1)
    units = contract.execution_contract.execution.units

    # 长文本，长度 > 300
    fallback_text = "降级成功文本。" + "这是一段很长的描述，用于确保场景文本长度超过300个字符的验证要求，从而使降级路径能够成功返回。这里我们填充足够的文字，让长度超过300个字符。继续填充文字，直到达到要求。现在我们已经写了很多，应该超过了300字符。继续写一些无关紧要的内容，只要确保长度足够即可。测试应该通过，因为降级路径会检测到这个长文本并通过验证。好了，现在长度肯定够了。再添加一些内容以确保万无一失。" * 3

    prompt_build_calls = []
    def mock_build_prompt(**kwargs):
        prompt_build_calls.append(kwargs)
        if kwargs.get('is_fallback', False):
            return "Fallback prompt"
        return "Normal prompt"

    async def mock_call_llm(prompt, max_tokens=None):
        if "Fallback prompt" in prompt:
            return f'{{"scene_text": "{fallback_text}", "events": []}}', {}
        return '{"scene_text": "失败文本", "events": []}', {}

    def mock_parse(text):
        # 如果是 fallback 的响应（包含长文本），返回长文本
        if "降级成功" in text:
            return MagicMock(scene_text=fallback_text, events=[])
        try:
            data = json.loads(text)
            return MagicMock(scene_text=data.get("scene_text", ""), events=data.get("events", []))
        except:
            return MagicMock(scene_text=text, events=[])

    with patch.object(writer, '_call_llm', side_effect=mock_call_llm):
        with patch.object(writer, '_parse_and_validate', side_effect=mock_parse):
            with patch.object(writer, '_build_segment_prompt', side_effect=mock_build_prompt):
                text, events, ok = await writer._execute_segment(
                    contract=contract, units=units, idx=0, total=1,
                    previous_text="", previous_events=[], current_state={}
                )

    assert ok is True, "Fallback should return True"
    assert "降级成功" in text or "降级成功文本" in text
    fallback_calls = [c for c in prompt_build_calls if c.get('is_fallback', False)]
    assert len(fallback_calls) >= 1, "Fallback prompt should be built"


@pytest.mark.asyncio
async def test_state_accumulation_across_segments():
    validator_mock = MagicMock(spec=SemanticValidator)
    validator_mock.validate.return_value = create_validation_result(True)
    writer = ControlledWriter(
        max_retries_per_segment=2,
        semantic_validator=validator_mock,
        quality_gate=QualityGate(),
    )
    contract = create_contract_with_units(2)
    units = contract.execution_contract.execution.units

    previous_events = [{"type": "item_acquire", "actor": "林逸", "item": "古玉"}]
    prompt_calls = []
    def mock_build_prompt(**kwargs):
        prompt_calls.append(kwargs)
        return "prompt"

    async def mock_call_llm(prompt, max_tokens=None):
        return '{"scene_text": "生成的文本", "events": []}', {}

    def mock_parse(text):
        return MagicMock(scene_text=text, events=[])

    with patch.object(writer, '_call_llm', side_effect=mock_call_llm):
        with patch.object(writer, '_parse_and_validate', side_effect=mock_parse):
            with patch.object(writer, '_build_segment_prompt', side_effect=mock_build_prompt):
                text, events, ok = await writer._execute_segment(
                    contract=contract, units=units, idx=0, total=1,
                    previous_text="", previous_events=previous_events, current_state={}
                )

    assert ok is True
    passed_events = [p.get('previous_events', []) for p in prompt_calls]
    assert any("古玉" in str(pe) for pe in passed_events)


@pytest.mark.asyncio
async def test_passes_on_first_success():
    validator_mock = MagicMock(spec=SemanticValidator)
    validator_mock.validate.return_value = create_validation_result(True)
    writer = ControlledWriter(
        max_retries_per_segment=2,
        semantic_validator=validator_mock,
        quality_gate=QualityGate(),
    )
    contract = create_contract_with_units(1)
    units = contract.execution_contract.execution.units

    prompt_calls = []
    async def mock_call_llm(prompt, max_tokens=None):
        return '{"scene_text": "第一次成功", "events": []}', {}

    def mock_parse(text):
        return MagicMock(scene_text=text, events=[])

    def mock_build_prompt(**kwargs):
        prompt_calls.append(kwargs)
        return "prompt"

    with patch.object(writer, '_call_llm', side_effect=mock_call_llm):
        with patch.object(writer, '_parse_and_validate', side_effect=mock_parse):
            with patch.object(writer, '_build_segment_prompt', side_effect=mock_build_prompt):
                writer.enable_fallback = False
                text, events, ok = await writer._execute_segment(
                    contract=contract, units=units, idx=0, total=1,
                    previous_text="", previous_events=[], current_state={}
                )

    assert ok is True
    assert "第一次成功" in text
    assert len(prompt_calls) == 1
    assert prompt_calls[0].get('error_hint', '') == ''
    assert validator_mock.validate.call_count == 1


@pytest.mark.asyncio
async def test_validation_result_contains_blocking_missing():
    contract = PlanningContract(
        version="1.0",
        scene_id="test_blocking",
        intent=Intent(goal="测试目标", conflict="测试冲突", expected_outcome="测试结果"),
        execution=Execution(units=[
            ExecutionUnit(id="U1", label="action", description="获得玉佩")
        ]),
        observables=Observables(state_changes=[
            StateChange(
                id="evt_001",
                type="plot_flag",
                source=SignalSource.LLM,
                name="获得玉佩",
                value=True,
            )
        ]),
        constraints=[],
        metadata=ContractMetadata(chapter=1, scene_index=0),
    )

    text = "林逸修炼了一整天，但没有得到任何宝物。"
    validator = SemanticValidator(embedding_provider=NoOpEmbeddingProvider())
    result = validator.validate(contract, text)

    assert result.passed is False
    assert "获得玉佩" in result.missing
    assert "获得玉佩" in result.blocking_missing
    assert result.blocking_missing_count == 1


@pytest.mark.asyncio
async def test_unknown_source_does_not_block():
    contract = PlanningContract(
        version="1.0",
        scene_id="test_unknown",
        intent=Intent(goal="测试目标", conflict="测试冲突", expected_outcome="测试结果"),
        execution=Execution(units=[
            ExecutionUnit(id="U1", label="action", description="获得玉佩")
        ]),
        observables=Observables(state_changes=[
            StateChange(
                id="evt_001",
                type="plot_flag",
                source=SignalSource.UNKNOWN,
                name="旧版遗留标记",
                value=True,
            )
        ]),
        constraints=[],
        metadata=ContractMetadata(chapter=1, scene_index=0),
    )

    text = "林逸修炼了一整天。"
    validator = SemanticValidator(embedding_provider=NoOpEmbeddingProvider())
    result = validator.validate(contract, text)

    assert result.passed is True
    assert "旧版遗留标记" in result.missing
    assert len(result.blocking_missing) == 0


@pytest.mark.asyncio
async def test_end_to_end_with_real_validator():
    validator = SemanticValidator(embedding_provider=NoOpEmbeddingProvider())
    writer = ControlledWriter(
        max_retries_per_segment=2,
        semantic_validator=validator,
        quality_gate=QualityGate(),
    )
    contract = create_contract_with_units(1)
    planning = contract.execution_contract
    planning.observables.state_changes.append(
        StateChange(
            id="evt_test",
            type="plot_flag",
            source=SignalSource.LLM,
            name="事件 1",
            value=True,
        )
    )
    units = planning.execution.units

    async def mock_call_llm(prompt, max_tokens=None):
        return '{"scene_text": "林逸完成了事件 1", "events": [{"type": "plot_flag_set", "flag": "事件 1", "value": true}]}', {}

    def mock_parse(text):
        return MagicMock(scene_text=text, events=[{"type": "plot_flag_set", "flag": "事件 1", "value": True}])

    with patch.object(writer, '_call_llm', side_effect=mock_call_llm):
        with patch.object(writer, '_parse_and_validate', side_effect=mock_parse):
            writer.enable_fallback = False
            text, events, ok = await writer._execute_segment(
                contract=contract, units=units, idx=0, total=1,
                previous_text="", previous_events=[], current_state={}
            )

    assert ok is True
    assert "事件 1" in text or "完成" in text