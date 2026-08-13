# tests/unit/writing/audit/test_preservation.py

import pytest
from src.writing.audit import (
    TraceCollector,
    PayloadRef,
    MemoryPayloadResolver,
    PreservationAnalyzer,
)
from src.writing.audit.field_comparator import Existence, ChangeType


def test_preservation_basic():
    resolver = MemoryPayloadResolver()
    with TraceCollector("novel", 1, 1, 0) as collector:
        plan_ref = PayloadRef("memory://planning")
        resolver.register(plan_ref, {"goal": "write", "must_events": ["A", "B", "C"]})
        plan_id = collector.record_reference("planning", plan_ref, "d1", 100)
        with collector.stage("planning") as s:
            s.output(plan_id)

        prompt_ref = PayloadRef("memory://prompt")
        resolver.register(prompt_ref, {"goal": "write", "must_events": ["A"]})
        prompt_id = collector.record_reference("prompt_bundle", prompt_ref, "d2", 200)
        with collector.stage("prompt") as s:
            s.input(plan_id).output(prompt_id)

        trace = collector.finish()

    analyzer = PreservationAnalyzer(resolver, fields=["goal", "must_events"])
    report = analyzer.analyze(trace)

    assert report.total_fields == 2
    goal_fp = report.fields["goal"]
    assert goal_fp.is_fully_preserved
    assert goal_fp.end_retention_rate == 1.0

    must_fp = report.fields["must_events"]
    assert not must_fp.is_fully_preserved
    # 3 → 1 保留，保留率 1/3
    assert must_fp.end_retention_rate == 1/3
    assert len(must_fp.lineages) > 0
    # first_absent_artifact 应当为 None，因为字段未完全消失（只是部分保留）
    for lp in must_fp.lineages:
        assert lp.first_absent_artifact is None