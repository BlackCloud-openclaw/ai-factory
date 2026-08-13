# tests/phase12/test_evaluation_snapshot.py

import pytest
from src.writing.evaluation import (
    EvaluationSnapshot,
    RuntimeMetrics,
    RevisionResult,
    JudgeContext,
)


def test_revision_delta():
    """验证 RevisionResult.delta 计算正确"""
    rev = RevisionResult(
        before_compliance=0.5,
        after_compliance=0.8,
    )
    # 使用 approx 处理浮点数精度
    assert rev.delta == pytest.approx(0.3)

    rev_negative = RevisionResult(
        before_compliance=0.9,
        after_compliance=0.6,
    )
    assert rev_negative.delta == pytest.approx(-0.3)


def test_snapshot_to_dict():
    """验证序列化"""
    snapshot = EvaluationSnapshot(
        scene_before="before_text",
        scene_after="after_text",
        runtime_metrics=RuntimeMetrics(
            retry_count=2,
            fallback_count=1,
            error_count=0,
            validation_score=0.95,
            execution_time_ms=1500,
            llm_calls=3,
            total_tokens=1200,
        ),
        revision_result=RevisionResult(
            before_compliance=0.5,
            after_compliance=0.8,
        ),
        judge_context=JudgeContext(
            previous_scene_text="prev",
            character_summary="char info",
            world_summary="world info",
        ),
        artifacts={"key": "value"},
    )

    d = snapshot.to_dict()

    assert d["scene_before"] == "before_text"
    assert d["scene_after"] == "after_text"
    assert d["runtime_metrics"]["retry_count"] == 2
    assert d["runtime_metrics"]["execution_time_ms"] == 1500
    assert d["revision_result"]["before_compliance"] == 0.5
    # 使用 approx 处理浮点数精度
    assert d["revision_result"]["delta"] == pytest.approx(0.3)
    assert d["judge_context"]["previous_scene_text"] == "prev"
    assert d["artifacts"]["key"] == "value"