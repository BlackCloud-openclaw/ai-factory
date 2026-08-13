# tests/unit/writing/audit/test_reporter.py

import pytest
from src.writing.audit import (
    TraceCollector,
    PayloadRef,
    MemoryPayloadResolver,
    PreservationAnalyzer,
    AttributionAnalyzer,
    MetricBudgetAnalyzer,
    PriorityEngine,
    Reporter,
)


def test_reporter_basic():
    resolver = MemoryPayloadResolver()

    with TraceCollector("novel", 1, 1, 0) as collector:
        plan_ref = PayloadRef("mem://planning")
        resolver.register(plan_ref, {"goal": "write", "outcome": "success", "must_events": ["A", "B", "C"]})
        plan_id = collector.record_reference("planning", plan_ref, "d1", 100)
        with collector.stage("planning") as s:
            s.output(plan_id).metric("tokens", 50)

        prompt_ref = PayloadRef("mem://prompt")
        resolver.register(prompt_ref, {"goal": "write", "must_events": ["A"]})
        prompt_id = collector.record_reference("prompt_bundle", prompt_ref, "d2", 200)
        with collector.stage("prompt") as s:
            s.input(plan_id).output(prompt_id).metric("tokens", 150)

        draft_ref = PayloadRef("mem://draft")
        resolver.register(draft_ref, {"goal": "write"})
        draft_id = collector.record_reference("draft", draft_ref, "d3", 300)
        with collector.stage("draft") as s:
            s.input(prompt_id).output(draft_id).metric("tokens", 100)

        trace = collector.finish()

    preservation = PreservationAnalyzer(resolver, fields=["goal", "outcome", "must_events"])
    pres_report = preservation.analyze(trace)

    attribution = AttributionAnalyzer(resolver)
    attr_report = attribution.analyze(trace, pres_report)

    budget = MetricBudgetAnalyzer()
    budget_report = budget.analyze(trace)

    engine = PriorityEngine()
    priority_report = engine.analyze(
        execution_id=str(trace.execution_id),
        preservation_report=pres_report,
        attribution_report=attr_report,
        budget_report=budget_report,
    )

    reporter = Reporter()
    comprehensive = reporter.generate(
        preservation_report=pres_report,
        attribution_report=attr_report,
        budget_report=budget_report,
        priority_report=priority_report,
    )

    assert comprehensive.execution_id == str(trace.execution_id)
    assert comprehensive.summary["execution_id"] == str(trace.execution_id)
    assert comprehensive.summary["total_fields"] == 3
    assert comprehensive.summary["fields_with_loss"] >= 1


def test_reporter_to_markdown():
    resolver = MemoryPayloadResolver()

    with TraceCollector("novel", 1, 1, 0) as collector:
        plan_ref = PayloadRef("mem://planning")
        resolver.register(plan_ref, {"goal": "write", "outcome": "success"})
        plan_id = collector.record_reference("planning", plan_ref, "d1", 100)
        with collector.stage("planning") as s:
            s.output(plan_id).metric("tokens", 50)

        prompt_ref = PayloadRef("mem://prompt")
        resolver.register(prompt_ref, {"goal": "write"})
        prompt_id = collector.record_reference("prompt_bundle", prompt_ref, "d2", 200)
        with collector.stage("prompt") as s:
            s.input(plan_id).output(prompt_id).metric("tokens", 150)

        trace = collector.finish()

    preservation = PreservationAnalyzer(resolver, fields=["outcome"])
    pres_report = preservation.analyze(trace)
    attribution = AttributionAnalyzer(resolver)
    attr_report = attribution.analyze(trace, pres_report)
    budget = MetricBudgetAnalyzer()
    budget_report = budget.analyze(trace)
    engine = PriorityEngine()
    priority_report = engine.analyze(
        execution_id=str(trace.execution_id),
        preservation_report=pres_report,
        attribution_report=attr_report,
        budget_report=budget_report,
    )

    reporter = Reporter()
    comprehensive = reporter.generate(
        preservation_report=pres_report,
        attribution_report=attr_report,
        budget_report=budget_report,
        priority_report=priority_report,
    )

    markdown = comprehensive.to_markdown()
    assert "Comprehensive Audit Report" in markdown
    assert "## Summary" in markdown
    assert "## 1. Preservation Analysis" in markdown
    assert "## 2. Attribution Analysis" in markdown
    assert "## 3. Budget Analysis" in markdown
    assert "## 4. Priority Recommendations" in markdown
    assert "outcome" in markdown


def test_reporter_to_dict():
    resolver = MemoryPayloadResolver()

    with TraceCollector("novel", 1, 1, 0) as collector:
        plan_ref = PayloadRef("mem://planning")
        resolver.register(plan_ref, {"goal": "write"})
        plan_id = collector.record_reference("planning", plan_ref, "d1", 100)
        with collector.stage("planning") as s:
            s.output(plan_id).metric("tokens", 50)

        trace = collector.finish()

    preservation = PreservationAnalyzer(resolver, fields=["goal"])
    pres_report = preservation.analyze(trace)
    attribution = AttributionAnalyzer(resolver)
    attr_report = attribution.analyze(trace, pres_report)
    budget = MetricBudgetAnalyzer()
    budget_report = budget.analyze(trace)
    engine = PriorityEngine()
    priority_report = engine.analyze(
        execution_id=str(trace.execution_id),
        preservation_report=pres_report,
        attribution_report=attr_report,
        budget_report=budget_report,
    )

    reporter = Reporter()
    comprehensive = reporter.generate(
        preservation_report=pres_report,
        attribution_report=attr_report,
        budget_report=budget_report,
        priority_report=priority_report,
    )

    data = comprehensive.to_dict()
    assert "execution_id" in data
    assert "summary" in data
    assert "preservation" in data
    assert "attribution" in data
    assert "budget" in data
    assert "priority" in data


def test_reporter_to_console():
    resolver = MemoryPayloadResolver()

    with TraceCollector("novel", 1, 1, 0) as collector:
        plan_ref = PayloadRef("mem://planning")
        resolver.register(plan_ref, {"outcome": "success"})
        plan_id = collector.record_reference("planning", plan_ref, "d1", 100)
        with collector.stage("planning") as s:
            s.output(plan_id).metric("tokens", 50)

        prompt_ref = PayloadRef("mem://prompt")
        resolver.register(prompt_ref, {})
        prompt_id = collector.record_reference("prompt_bundle", prompt_ref, "d2", 200)
        with collector.stage("prompt") as s:
            s.input(plan_id).output(prompt_id).metric("tokens", 150)

        trace = collector.finish()

    preservation = PreservationAnalyzer(resolver, fields=["outcome"])
    pres_report = preservation.analyze(trace)
    attribution = AttributionAnalyzer(resolver)
    attr_report = attribution.analyze(trace, pres_report)
    budget = MetricBudgetAnalyzer()
    budget_report = budget.analyze(trace)
    engine = PriorityEngine()
    priority_report = engine.analyze(
        execution_id=str(trace.execution_id),
        preservation_report=pres_report,
        attribution_report=attr_report,
        budget_report=budget_report,
    )

    reporter = Reporter()
    comprehensive = reporter.generate(
        preservation_report=pres_report,
        attribution_report=attr_report,
        budget_report=budget_report,
        priority_report=priority_report,
    )

    console = comprehensive.to_console()
    assert "Report:" in console
    assert "Fields:" in console
    assert "Priority:" in console


def test_reporter_verifies_execution_id():
    resolver = MemoryPayloadResolver()

    with TraceCollector("novel", 1, 1, 0) as collector:
        plan_ref = PayloadRef("mem://planning")
        resolver.register(plan_ref, {"goal": "write"})
        plan_id = collector.record_reference("planning", plan_ref, "d1", 100)
        with collector.stage("planning") as s:
            s.output(plan_id).metric("tokens", 50)

        trace = collector.finish()

    preservation = PreservationAnalyzer(resolver, fields=["goal"])
    pres_report = preservation.analyze(trace)

    # 创建 execution_id 不匹配的报告
    from src.writing.audit.attribution import AttributionReport
    bad_attr_report = AttributionReport(
        execution_id="mismatch",
        total_fields_analyzed=0,
        fields_with_loss=0,
        fields_without_loss=0,
        attributions={},
        by_type={},
    )

    budget = MetricBudgetAnalyzer()
    budget_report = budget.analyze(trace)

    engine = PriorityEngine()
    priority_report = engine.analyze(
        execution_id=str(trace.execution_id),
        preservation_report=pres_report,
        attribution_report=bad_attr_report,
        budget_report=budget_report,
    )

    reporter = Reporter()
    with pytest.raises(ValueError, match="execution_id mismatch"):
        reporter.generate(
            preservation_report=pres_report,
            attribution_report=bad_attr_report,
            budget_report=budget_report,      # ✅ 修正
            priority_report=priority_report,  # ✅ 修正
        )