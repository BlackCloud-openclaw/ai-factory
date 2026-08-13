# tests/unit/writing/audit/test_priority.py

import pytest
from src.writing.audit import (
    TraceCollector,
    PayloadRef,
    MemoryPayloadResolver,
    PreservationAnalyzer,
    AttributionAnalyzer,
    MetricBudgetAnalyzer,
    PriorityEngine,
    PriorityLevel,
    AttributionType,
    PriorityPolicy,
    OptimizationTarget,          # 添加这一行
)


def test_priority_engine_basic():
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

    assert priority_report.total_targets >= 2

    assert priority_report.targets
    first = priority_report.targets[0]
    assert first.field_name in ["outcome", "must_events"]
    assert first.severity in [PriorityLevel.CRITICAL, PriorityLevel.HIGH]


def test_priority_engine_no_loss():
    resolver = MemoryPayloadResolver()

    with TraceCollector("novel", 1, 1, 0) as collector:
        plan_ref = PayloadRef("mem://planning")
        resolver.register(plan_ref, {"goal": "write"})
        plan_id = collector.record_reference("planning", plan_ref, "d1", 100)
        with collector.stage("planning") as s:
            s.output(plan_id).metric("tokens", 50)

        prompt_ref = PayloadRef("mem://prompt")
        resolver.register(prompt_ref, {"goal": "write"})
        prompt_id = collector.record_reference("prompt_bundle", prompt_ref, "d2", 200)
        with collector.stage("prompt") as s:
            s.input(plan_id).output(prompt_id).metric("tokens", 150)

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

    assert priority_report.total_targets == 0


def test_priority_to_markdown():
    resolver = MemoryPayloadResolver()

    with TraceCollector("novel", 1, 1, 0) as collector:
        plan_ref = PayloadRef("mem://planning")
        resolver.register(plan_ref, {"outcome": "success", "goal": "write"})
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

    markdown = priority_report.to_markdown()
    assert "Priority Report" in markdown
    assert "outcome" in markdown
    assert "Top Recommendation" in markdown


def test_priority_factors():
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

    engine = PriorityEngine(
        policy=PriorityPolicy(
            retention_weight=0.5,
            stage_score_weight=0.2,
            stage_position_weight=0.2,
            attribution_weight=0.1,
        )
    )
    priority_report = engine.analyze(
        execution_id=str(trace.execution_id),
        preservation_report=pres_report,
        attribution_report=attr_report,
        budget_report=budget_report,
    )

    if priority_report.total_targets > 0:
        first = priority_report.targets[0]
        assert len(first.factors) == 4
        total_contrib = sum(f.contribution for f in first.factors)
        assert 0.0 <= total_contrib <= 1.0


def test_priority_get_by_severity():
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

    criticals = priority_report.get_by_severity(PriorityLevel.CRITICAL)
    assert isinstance(criticals, tuple)


def test_priority_top_critical_property():
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

    top = priority_report.top_critical
    assert top is None or isinstance(top, OptimizationTarget)


def test_priority_unknown_stage_name():
    """测试未知阶段名称的安全处理。"""
    from src.writing.stage_names import StageName
    assert StageName.safe_parse("unknown_stage") is None
    assert StageName.safe_parse("planning") == StageName.PLANNING