import pytest
import json
import yaml
from pathlib import Path

from experiments.phase12.factory.context_factory import create_context_from_sample
from experiments.phase12.benchmark import BenchmarkRunner
from experiments.phase12.metrics import MetricRegistry
from experiments.phase12.model import BenchmarkResult
from experiments.phase12.config.benchmark import (
    GOLDEN_TOLERANCE,
    JUDGE_GOLDEN_TOLERANCE,
    GOLDEN_DETERMINISTIC_PATH,
    GOLDEN_JUDGE_PATH,
    BENCHMARK_VERSION,
    JUDGE_MODEL,
)
from experiments.phase12.judge.prompts import PROMPT_VERSIONS

import logging

logging.basicConfig(level=logging.INFO)

def sample_contexts():
    corpus_path = Path("experiments/phase12/corpus/v1.0/corpus.yaml")
    if not corpus_path.exists():
        pytest.skip("Corpus file not found")

    print(f"Loading corpus from: {corpus_path}")
    with open(corpus_path, "r") as f:
        manifest = yaml.safe_load(f)

    print(f"Manifest entries: {len(manifest.get('samples', []))}")
    base_dir = corpus_path.parent
    contexts = []
    for entry in manifest.get("samples", []):
        sample_path = base_dir / entry["path"]
        print(f"Processing: {sample_path}")
        if not sample_path.exists():
            print(f"  ⚠️ File not found")
            continue
        with open(sample_path, "r") as sf:
            sample_data = yaml.safe_load(sf)
        try:
            ctx = create_context_from_sample(sample_data)
            contexts.append(ctx)
            print(f"  ✅ Created context")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"Total contexts: {len(contexts)}")
    return contexts

def is_deterministic_metric(name: str) -> bool:
    deterministic = ["planning_coverage", "state_consistency", "runtime_health", "revision_pass_rate"]
    return name in deterministic


@pytest.fixture
def sample_contexts():
    """从 corpus.yaml 加载所有样本"""
    corpus_path = Path("experiments/phase12/corpus/v1.0/corpus.yaml")
    if not corpus_path.exists():
        pytest.skip("Corpus file not found, skipping integration test")

    with open(corpus_path, "r") as f:
        manifest = yaml.safe_load(f)

    base_dir = corpus_path.parent
    contexts = []
    for entry in manifest.get("samples", []):
        sample_path = base_dir / entry["path"]
        if not sample_path.exists():
            continue
        with open(sample_path, "r") as sf:
            sample_data = yaml.safe_load(sf)
        ctx = create_context_from_sample(sample_data)
        contexts.append(ctx)
    return contexts


@pytest.mark.asyncio
async def test_full_benchmark_pipeline(sample_contexts):
    print(f"📦 sample_contexts length: {len(sample_contexts)}")
    registry = MetricRegistry.with_defaults()
    runner = BenchmarkRunner(registry)
    result = await runner.run(sample_contexts)
    
    print(f"🔢 total_tasks: {result.metadata.get('total_tasks')}")
    print(f"📊 metrics_count: {result.metadata.get('metrics_count')}")
    print(f"📋 metrics: {result.metadata.get('metrics')}")
    
    for mr in result.metric_results:
        print(f"  {mr.name}: sample_count={mr.details.get('sample_count')}, score={mr.score}")

    assert "metrics" in result.metadata
    assert "benchmark_version" in result.metadata
    assert "error_count" in result.metadata
    assert len(result.metadata["metrics"]) == 8

    result_path = Path("experiments/phase12/results/latest.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    report_path = Path("experiments/phase12/results/latest.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(_to_markdown(result))


def _to_markdown(result: BenchmarkResult) -> str:
    lines = [
        "# Benchmark Report",
        "",
        f"**Overall Score:** {result.overall_score:.3f}",
        f"**Version:** {result.metadata.get('benchmark_version', 'unknown')}",
        f"**Success/Error:** {result.metadata.get('success_count', 0)}/{result.metadata.get('error_count', 0)}",
        "",
        "## Metric Results",
        "",
        "| Metric | Score | State | Passed | Details |",
        "|--------|-------|-------|--------|---------|",
    ]
    for mr in result.metric_results:
        score = f"{mr.score:.3f}" if mr.score is not None else "N/A"
        state = mr.state.value
        passed = "✅" if mr.passed else "❌"
        details = ", ".join(f"{k}={v}" for k, v in mr.details.items() if isinstance(v, (int, float, str)))
        lines.append(f"| {mr.name} | {score} | {state} | {passed} | {details} |")
    return "\n".join(lines)


@pytest.mark.asyncio
async def test_golden_baseline_deterministic(sample_contexts):
    registry = MetricRegistry.with_defaults()
    runner = BenchmarkRunner(registry)
    result = await runner.run(sample_contexts)

    deterministic_results = [m for m in result.metric_results if is_deterministic_metric(m.name)]
    if not deterministic_results:
        pytest.skip("No deterministic metrics found")

    golden_path = Path(GOLDEN_DETERMINISTIC_PATH)
    if not golden_path.exists():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        baseline = {
            "benchmark_version": BENCHMARK_VERSION,
            "overall_score": result.overall_score,
            "metric_scores": {m.name: m.score for m in deterministic_results},
        }
        with open(golden_path, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2, ensure_ascii=False)
        pytest.skip("Deterministic golden baseline created, run again to verify")

    with open(golden_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    if baseline.get("benchmark_version") != BENCHMARK_VERSION:
        pytest.skip(f"Version mismatch: {baseline.get('benchmark_version')} vs {BENCHMARK_VERSION}")

    assert abs(result.overall_score - baseline.get("overall_score", 0.0)) < GOLDEN_TOLERANCE
    for m in deterministic_results:
        expected = baseline.get("metric_scores", {}).get(m.name)
        if expected is not None:
            assert abs(m.score - expected) < GOLDEN_TOLERANCE


@pytest.mark.asyncio
async def test_golden_baseline_judge(sample_contexts):
    registry = MetricRegistry.with_defaults()
    runner = BenchmarkRunner(registry)
    result = await runner.run(sample_contexts)

    judge_results = [m for m in result.metric_results if not is_deterministic_metric(m.name)]
    valid_judge = [m for m in judge_results if m.score is not None]
    if not valid_judge:
        pytest.skip("No valid Judge scores (LLM may not be running or context missing)")

    golden_path = Path(GOLDEN_JUDGE_PATH)
    if not golden_path.exists():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        baseline = {
            "benchmark_version": BENCHMARK_VERSION,
            "overall_score": result.overall_score,
            "model": JUDGE_MODEL,
            "prompt_versions": {k.value: v for k, v in PROMPT_VERSIONS.items()},
            "metric_scores": {m.name: m.score for m in valid_judge},
        }
        with open(golden_path, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2, ensure_ascii=False)
        pytest.skip("Judge golden baseline created, run again to verify")

    with open(golden_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    if baseline.get("benchmark_version") != BENCHMARK_VERSION:
        pytest.skip(f"Version mismatch: {baseline.get('benchmark_version')} vs {BENCHMARK_VERSION}")
    if baseline.get("model") != JUDGE_MODEL:
        pytest.skip(f"Model mismatch: {baseline.get('model')} vs {JUDGE_MODEL}")
    for dim in PROMPT_VERSIONS:
        expected_version = baseline.get("prompt_versions", {}).get(dim.value)
        if expected_version != PROMPT_VERSIONS[dim]:
            pytest.skip(f"Prompt version mismatch for {dim.value}: {expected_version} vs {PROMPT_VERSIONS[dim]}")

    assert abs(result.overall_score - baseline.get("overall_score", 0.0)) < JUDGE_GOLDEN_TOLERANCE
    for m in valid_judge:
        expected = baseline.get("metric_scores", {}).get(m.name)
        if expected is not None:
            assert abs(m.score - expected) < JUDGE_GOLDEN_TOLERANCE