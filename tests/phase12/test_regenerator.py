"""
测试 CorpusRegenerator（使用 Mock Writer）
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from src.writing.evaluation import (
    EvaluationSnapshot,
    RuntimeMetrics,
    RevisionResult,
    JudgeContext,
)
from experiments.phase12.corpus.models import CorpusSample, FailureMode, Difficulty
from experiments.phase12.corpus.regenerator import CorpusRegenerator, RegenerateResult


def create_test_sample() -> CorpusSample:
    return CorpusSample(
        id="test_sample_001",
        version="1.0",
        category="runtime_state",
        failure_modes=["runtime_state"],
        difficulty=Difficulty.MEDIUM,
        language="zh-CN",
        scene_before="林逸站在秘境入口，灵力紊乱。",
        scene_after=None,
        draft_before=None,
        draft_after=None,
        expected={},
        artifacts={},
        source="test",
        license="internal",
        tags=(),
    )


@pytest.mark.asyncio
async def test_regenerate_sample_success(tmp_path):
    """测试单样本再生成功"""
    # 创建 Mock Writer
    mock_writer = AsyncMock()
    mock_writer.execute_with_snapshot.return_value = EvaluationSnapshot(
        scene_before="林逸站在秘境入口，灵力紊乱。",
        scene_after="林逸踏入秘境，灵力逐渐平复。",
        runtime_metrics=RuntimeMetrics(
            retry_count=0,
            fallback_count=0,
            error_count=0,
            validation_score=0.95,
            execution_time_ms=500,
        ),
        revision_result=RevisionResult(
            before_compliance=0.6,
            after_compliance=0.9,
        ),
        judge_context=JudgeContext(
            previous_scene_text="上一场景",
            character_summary='{"林逸": {"hp": 100}}',
            world_summary='{"location": "秘境"}',
        ),
    )

    regenerator = CorpusRegenerator(
        writer=mock_writer,
        output_dir=tmp_path,
        version="2.0",
    )

    sample = create_test_sample()
    result = await regenerator.regenerate_sample(
        sample=sample,
        category="runtime_state",
        novel_id="test_novel",
        chapter=1,
        scene_idx=0,
    )

    assert result.success is True
    assert result.sample_id == "test_sample_001"
    assert result.category == "runtime_state"
    assert result.output_path.exists()
    assert result.output_path.suffix == ".yaml"

    # 验证 Writer 被正确调用
    mock_writer.execute_with_snapshot.assert_called_once()


@pytest.mark.asyncio
async def test_regenerate_sample_failure(tmp_path):
    """测试单样本再生失败"""
    mock_writer = AsyncMock()
    mock_writer.execute_with_snapshot.side_effect = Exception("Writer 执行失败")

    regenerator = CorpusRegenerator(
        writer=mock_writer,
        output_dir=tmp_path,
    )

    sample = create_test_sample()
    result = await regenerator.regenerate_sample(sample)

    assert result.success is False
    assert "Writer 执行失败" in result.error


@pytest.mark.asyncio
async def test_regenerator_output_structure(tmp_path):
    """验证输出 YAML 结构符合 v2.0 规范"""
    mock_writer = AsyncMock()
    mock_writer.execute_with_snapshot.return_value = EvaluationSnapshot(
        scene_before="before",
        scene_after="after",
        runtime_metrics=RuntimeMetrics(
            retry_count=1,
            fallback_count=0,
            error_count=0,
            validation_score=0.9,
            execution_time_ms=1000,
        ),
        revision_result=RevisionResult(
            before_compliance=0.5,
            after_compliance=0.8,
        ),
        judge_context=JudgeContext(
            previous_scene_text="prev",
            character_summary='{"char": "info"}',
            world_summary='{"world": "info"}',
        ),
    )

    regenerator = CorpusRegenerator(
        writer=mock_writer,
        output_dir=tmp_path,
        version="2.0",
    )

    sample = create_test_sample()
    result = await regenerator.regenerate_sample(
        sample,
        category="scene_transition",
    )

    # 读取生成的 YAML 验证结构
    import yaml
    with open(result.output_path) as f:
        data = yaml.safe_load(f)

    assert data["version"] == "2.0"
    assert data["category"] == "scene_transition"
    assert data["artifacts"]["runtime_metrics"]["retry_count"] == 1
    assert data["artifacts"]["revision_result"]["before_compliance"] == 0.5
    assert data["artifacts"]["judge_context"]["previous_scene_text"] == "prev"