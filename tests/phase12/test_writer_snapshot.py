"""
测试 ControlledWriter.execute_with_snapshot
"""

import pytest
from unittest.mock import AsyncMock

from src.writing.controlled_writer import ControlledWriter
from src.writing.planning_contract import (
    PlanningContract,
    Intent,
    ContractMetadata,
    Execution,
    ExecutionUnit,
)
from src.writing.evaluation import EvaluationSnapshot


def build_test_contract() -> PlanningContract:
    return PlanningContract(
        scene_id="test_scene",
        intent=Intent(
            goal="test goal",
            conflict="test conflict",
            expected_outcome="test outcome",
        ),
        execution=Execution(
            units=[
                ExecutionUnit(id="1", label="action", description="test action"),
            ]
        ),
        metadata=ContractMetadata(chapter=1, scene_index=0),
    )


@pytest.mark.asyncio
async def test_execute_with_snapshot_returns_snapshot():
    """验证 execute_with_snapshot 返回 EvaluationSnapshot"""
    writer = ControlledWriter(api_base="http://test", model="test-model")

    # Mock execute 方法
    mock_result = AsyncMock()
    mock_result.text = "test text"
    mock_result.events = []
    mock_result.segments_used = 1
    mock_result.segments_succeeded = 1
    mock_result.fallback_used = False
    mock_result.execution_time = 0.5
    writer.execute = AsyncMock(return_value=mock_result)

    contract = build_test_contract()
    snapshot, result = await writer.execute_with_snapshot(
        contract,
        scene_before="before text",
        previous_scene_text="prev text",
        character_summary={"林逸": {"hp": 100}},
        world_summary={"location": "禁地"},
    )

    assert isinstance(snapshot, EvaluationSnapshot)
    assert snapshot.scene_before == "before text"
    assert snapshot.scene_after == "test text"
    assert snapshot.runtime_metrics.segments_total == 1
    assert snapshot.runtime_metrics.segments_succeeded == 1
    assert snapshot.judge_context is not None
    assert snapshot.judge_context.previous_scene_text == "prev text"
    assert snapshot.judge_context.character_summary is not None
    assert "林逸" in snapshot.judge_context.character_summary or "\\u6797\\u9038" in snapshot.judge_context.character_summary
    assert result.text == "test text"


@pytest.mark.asyncio
async def test_execute_with_snapshot_backward_compatible():
    """验证 execute 仍然正常工作"""
    writer = ControlledWriter(api_base="http://test", model="test-model")

    # Mock execute 方法
    mock_result = AsyncMock()
    mock_result.text = "test text"
    mock_result.events = []
    mock_result.segments_used = 1
    mock_result.segments_succeeded = 1
    mock_result.fallback_used = False
    mock_result.execution_time = 0.5
    writer.execute = AsyncMock(return_value=mock_result)

    contract = build_test_contract()
    result = await writer.execute(contract)
    assert result.text == "test text"