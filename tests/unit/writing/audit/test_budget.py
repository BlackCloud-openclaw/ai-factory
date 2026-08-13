# tests/unit/writing/audit/test_budget.py

import pytest
from src.writing.audit import (
    TraceCollector,
    PayloadRef,
    MetricBudgetAnalyzer,
    BudgetAnalyzer,
    StageMetricBudget,
    BudgetAnomaly,
    BudgetSeverity,
    BudgetAnomalyKind,
    MetricName,
)


def test_budget_analyzer_basic():
    with TraceCollector("novel", 1, 1, 0) as collector:
        plan_id = collector.record_reference("planning", PayloadRef("mem://planning"), "d1", 100)
        with collector.stage("planning") as s:
            s.output(plan_id).metric("tokens", 100)

        prompt_id = collector.record_reference("prompt_bundle", PayloadRef("mem://prompt"), "d2", 200)
        with collector.stage("prompt") as s:
            s.input(plan_id).output(prompt_id).metric("tokens", 300)

        draft_id = collector.record_reference("draft", PayloadRef("mem://draft"), "d3", 300)
        with collector.stage("draft") as s:
            s.input(prompt_id).output(draft_id).metric("tokens", 500)

        trace = collector.finish()

    analyzer = BudgetAnalyzer(metric=MetricName.TOKENS)
    report = analyzer.analyze(trace)

    assert report.metric == "tokens"
    assert report.total_metric_value == 900
    assert len(report.stages) == 3
    stages = {s.stage: s.metric_value for s in report.stages}
    assert stages["planning"] == 100
    assert stages["prompt"] == 300
    assert stages["draft"] == 500

    assert report.stages[0].stage == "draft"
    assert report.stages[0].metric_value == 500
    assert report.stages[0].percentage == 500/900
    assert report.stages[1].stage == "prompt"
    assert report.stages[2].stage == "planning"

    anomalies = [a for a in report.anomalies if a.kind == BudgetAnomalyKind.HIGH_USAGE]
    assert len(anomalies) == 1
    assert anomalies[0].stage == "draft"


def test_budget_analyzer_no_metrics():
    with TraceCollector("novel", 1, 1, 0) as collector:
        aid = collector.record_reference("planning", PayloadRef("mem://planning"), "d", 10)
        with collector.stage("planning") as s:
            s.output(aid)
        trace = collector.finish()

    analyzer = BudgetAnalyzer()
    report = analyzer.analyze(trace)
    assert report.total_metric_value == 0
    assert len(report.stages) == 0
    anomalies = [a for a in report.anomalies if a.kind == BudgetAnomalyKind.NO_DATA]
    assert len(anomalies) == 1


def test_budget_analyzer_invalid_metric():
    with TraceCollector("novel", 1, 1, 0) as collector:
        aid = collector.record_reference("planning", PayloadRef("mem://planning"), "d1", 10)
        with collector.stage("planning") as s:
            s.output(aid).metric("tokens", "abc")
        trace = collector.finish()

    analyzer = BudgetAnalyzer()
    report = analyzer.analyze(trace)
    assert report.total_metric_value == 0
    anomalies = [a for a in report.anomalies if a.kind == BudgetAnomalyKind.INVALID_VALUE]
    assert len(anomalies) == 1
    assert anomalies[0].stage == "planning"
    assert anomalies[0].raw_value == "abc"


def test_budget_analyzer_custom_metric():
    with TraceCollector("novel", 1, 1, 0) as collector:
        aid = collector.record_reference("planning", PayloadRef("mem://planning"), "d1", 10)
        with collector.stage("planning") as s:
            s.output(aid).metric("latency_ms", 10)

        aid2 = collector.record_reference("prompt_bundle", PayloadRef("mem://prompt"), "d2", 10)
        with collector.stage("prompt") as s:
            s.output(aid2).metric("latency_ms", 20)

        trace = collector.finish()

    analyzer = MetricBudgetAnalyzer(metric=MetricName.LATENCY_MS)
    report = analyzer.analyze(trace)
    assert report.metric == "latency_ms"
    assert report.total_metric_value == 30
    stages = {s.stage: s.metric_value for s in report.stages}
    assert stages["planning"] == 10
    assert stages["prompt"] == 20


def test_budget_analyzer_string_metric():
    with TraceCollector("novel", 1, 1, 0) as collector:
        aid = collector.record_reference("planning", PayloadRef("mem://planning"), "d1", 10)
        with collector.stage("planning") as s:
            s.output(aid).metric("custom_metric", 42)
        trace = collector.finish()

    analyzer = MetricBudgetAnalyzer(metric="custom_metric")
    report = analyzer.analyze(trace)
    assert report.metric == "custom_metric"
    assert report.total_metric_value == 42


def test_budget_analyzer_unknown_metric():
    with TraceCollector("novel", 1, 1, 0) as collector:
        aid = collector.record_reference("planning", PayloadRef("mem://planning"), "d1", 10)
        with collector.stage("planning") as s:
            s.output(aid).metric("unknown_metric", 100)
        trace = collector.finish()

    analyzer = MetricBudgetAnalyzer(metric="unknown_metric")
    report = analyzer.analyze(trace)
    assert report.total_metric_value == 100
    assert not any(a.kind == BudgetAnomalyKind.UNKNOWN_METRIC for a in report.anomalies)

    analyzer_strict = MetricBudgetAnalyzer(metric="unknown_metric", check_unknown_metric=True)
    report_strict = analyzer_strict.analyze(trace)
    anomalies = [a for a in report_strict.anomalies if a.kind == BudgetAnomalyKind.UNKNOWN_METRIC]
    assert len(anomalies) == 1
    assert anomalies[0].severity == BudgetSeverity.INFO
    assert report_strict.total_metric_value == 100


def test_budget_analyzer_to_markdown():
    with TraceCollector("novel", 1, 1, 0) as collector:
        aid = collector.record_reference("planning", PayloadRef("mem://planning"), "d1", 10)
        with collector.stage("planning") as s:
            s.output(aid).metric("tokens", 100)
        trace = collector.finish()

    analyzer = BudgetAnalyzer()
    report = analyzer.analyze(trace)
    markdown = report.to_markdown()
    assert "Budget Analysis Report" in markdown
    assert "**Total Value:** 100" in markdown
    assert "planning" in markdown
    # 修正：匹配实际格式（包含粗体标记）
    assert "**Metric:** `tokens`" in markdown


def test_budget_analyzer_stable_sort():
    with TraceCollector("novel", 1, 1, 0) as collector:
        a1 = collector.record_reference("planning", PayloadRef("mem://planning"), "d1", 10)
        with collector.stage("planning") as s:
            s.output(a1).metric("tokens", 100)

        a2 = collector.record_reference("draft", PayloadRef("mem://draft"), "d2", 20)
        with collector.stage("draft") as s:
            s.output(a2).metric("tokens", 100)

        trace = collector.finish()

    analyzer = BudgetAnalyzer()
    report = analyzer.analyze(trace)
    # 稳定排序：值相同则按 stage 升序
    assert report.stages[0].stage == "draft"
    assert report.stages[1].stage == "planning"