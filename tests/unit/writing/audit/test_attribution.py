# tests/unit/writing/audit/test_attribution.py

import pytest
from src.writing.audit import (
    TraceCollector,
    PayloadRef,
    MemoryPayloadResolver,
    PreservationAnalyzer,
    AttributionAnalyzer,
)
from src.writing.audit.field_comparator import Existence, ChangeType
from src.writing.audit.attribution import AttributionType


def test_attribution_transform_lost():
    resolver = MemoryPayloadResolver()
    with TraceCollector("novel", 1, 1, 0) as collector:
        plan_ref = PayloadRef("memory://planning")
        resolver.register(plan_ref, {"outcome": "success", "goal": "write"})
        plan_id = collector.record_reference("planning", plan_ref, "d1", 100)
        with collector.stage("planning") as s:
            s.output(plan_id)

        prompt_ref = PayloadRef("memory://prompt")
        resolver.register(prompt_ref, {"goal": "write"})  # outcome lost
        prompt_id = collector.record_reference("prompt_bundle", prompt_ref, "d2", 200)
        with collector.stage("prompt") as s:
            s.input(plan_id).output(prompt_id)

        trace = collector.finish()

    preservation = PreservationAnalyzer(resolver, fields=["outcome"])
    pres_report = preservation.analyze(trace)

    attribution = AttributionAnalyzer(resolver)
    attr_report = attribution.analyze(trace, pres_report)

    outcome_attr = attr_report.attributions["outcome"]
    assert outcome_attr.lost_stage == "prompt"
    assert outcome_attr.attribution_type == AttributionType.TRANSFORM_LOST
    assert outcome_attr.input_existence == Existence.PRESENT
    assert outcome_attr.output_existence == Existence.REMOVED