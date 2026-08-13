import pytest
import yaml
from pathlib import Path
from src.writing.evaluation import (
    EvaluationSnapshot,
    RuntimeMetrics,
    RevisionResult,
    JudgeContext,
)
from experiments.phase12.corpus.exporter import CorpusExporter


def test_export_snapshot(tmp_path):
    """测试导出 Snapshot 为 YAML v2.0"""
    snapshot = EvaluationSnapshot(
        scene_before="before text",
        scene_after="after text",
        runtime_metrics=RuntimeMetrics(
            retry_count=1,
            fallback_count=0,
            error_count=0,
            validation_score=0.95,
            execution_time_ms=1234,
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
        artifacts={"events": [{"type": "test"}]},
    )

    exporter = CorpusExporter(tmp_path, version="2.0")
    path = exporter.export(
        snapshot,
        category="dialogue_quality",
        failure_modes=["dialogue_quality"],
        sample_id="test_sample_001",
    )

    assert path.exists()
    assert path.suffix == ".yaml"

    # 验证内容
    with open(path) as f:
        data = yaml.safe_load(f)

    # 检查 v2.0 结构
    assert data["version"] == "2.0"
    assert data["category"] == "dialogue_quality"
    assert data["failure_modes"] == ["dialogue_quality"]
    assert data["scene_before"] == "before text"
    assert data["scene_after"] == "after text"

    # 检查 artifacts
    artifacts = data["artifacts"]
    assert artifacts["runtime_metrics"]["retry_count"] == 1
    assert artifacts["runtime_metrics"]["validation_score"] == 0.95
    assert artifacts["revision_result"]["before_compliance"] == 0.5
    assert artifacts["revision_result"]["delta"] == pytest.approx(0.3)
    assert artifacts["judge_context"]["previous_scene_text"] == "prev"
    assert artifacts["events"] == [{"type": "test"}]


def test_export_batch(tmp_path):
    """测试批量导出"""
    snapshots = [
        EvaluationSnapshot(
            scene_before=f"before {i}",
            scene_after=f"after {i}",
            runtime_metrics=RuntimeMetrics(retry_count=i),
        )
        for i in range(3)
    ]

    exporter = CorpusExporter(tmp_path)
    paths = exporter.export_batch(
        snapshots,
        categories=["scene_transition", "character_state", "dialogue_quality"],
    )

    assert len(paths) == 3
    assert all(p.exists() for p in paths)

    # 验证第一个文件的 category
    with open(paths[0]) as f:
        data = yaml.safe_load(f)
    assert data["category"] == "scene_transition"