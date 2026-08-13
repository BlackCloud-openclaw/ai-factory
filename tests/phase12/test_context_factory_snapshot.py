"""
测试 ContextFactory.from_snapshot
"""

import pytest
from src.writing.evaluation import (
    EvaluationSnapshot,
    RuntimeMetrics,
    RevisionResult,
    JudgeContext,
)
from experiments.phase12.corpus.factory import ContextFactory


def test_from_snapshot():
    snapshot = EvaluationSnapshot(
        scene_before="before",  # 这个字段只存在 snapshot 中
        scene_after="after",
        runtime_metrics=RuntimeMetrics(
            retry_count=1,
            fallback_count=0,
            error_count=0,
            validation_score=0.9,
            execution_time_ms=1234,
        ),
    )

    ctx = ContextFactory.from_snapshot(snapshot)

    # 改：用 scene_text 替代 scene_before
    assert ctx.scene_text == "after"  # 而不是 scene_before
    assert ctx.runtime_metrics.validation_score == 0.9
    assert ctx.runtime_metrics.execution_time_ms == 1234

def test_judge_context_preserved():
    """验证 JudgeContext 保留"""
    snapshot = EvaluationSnapshot(
        scene_before="before",
        scene_after="after",
        runtime_metrics=RuntimeMetrics(),
        judge_context=JudgeContext(
            previous_scene_text="previous",
            character_summary='{"林逸": {"hp": 100}}',
            world_summary='{"location": "禁地"}',
        ),
    )

    ctx = ContextFactory.from_snapshot(snapshot)

    assert ctx.judge_context is not None
    assert ctx.judge_context.previous_scene_text == "previous"
    assert ctx.judge_context.character_summary == '{"林逸": {"hp": 100}}'


def test_revision_preserved():
    """验证 RevisionResult 保留"""
    snapshot = EvaluationSnapshot(
        scene_before="before",
        scene_after="after",
        runtime_metrics=RuntimeMetrics(),
        revision_result=RevisionResult(
            before_compliance=0.5,
            after_compliance=0.9,
        ),
    )

    ctx = ContextFactory.from_snapshot(snapshot)

    assert ctx.revision_result is not None
    assert ctx.revision_result.before_compliance == 0.5
    assert ctx.revision_result.after_compliance == 0.9
    assert ctx.revision_result.delta == pytest.approx(0.4)


def test_nullable_fields():
    """验证可选字段为 None 时正常处理"""
    snapshot = EvaluationSnapshot(
        scene_before="before",
        scene_after="after",
        runtime_metrics=RuntimeMetrics(),
        revision_result=None,
        judge_context=None,
    )

    ctx = ContextFactory.from_snapshot(snapshot)

    assert ctx.revision_result is None
    assert ctx.judge_context is None