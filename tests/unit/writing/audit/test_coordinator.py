# tests/unit/writing/audit/test_coordinator.py

import pytest
from pathlib import Path
import tempfile

from src.writing.audit import (
    AuditCoordinator,
    AuditConfig,
    AuditReportStore,
    MemoryPayloadResolver,
    PayloadRef,
    TraceCollector,
)


def test_coordinator_basic():
    resolver = MemoryPayloadResolver()
    coordinator = AuditCoordinator(resolver=resolver)
    with coordinator.audit("novel", 1, 1, 0) as ctx:
        if ctx.collector:
            plan_ref = PayloadRef("memory://planning/001")
            plan_id = ctx.collector.record_reference("planning", plan_ref, "digest", 100)
            ctx.collector.record_stage("planning", outputs={"plan_id": plan_id})

    assert ctx.execution_id is not None


def test_coordinator_disabled():
    config = AuditConfig(enabled=False)
    coordinator = AuditCoordinator(config=config)
    with coordinator.audit("novel", 1, 1, 0) as ctx:
        pass

    assert ctx.report is None
    assert ctx.trace is None


def test_coordinator_record_stage():
    resolver = MemoryPayloadResolver()
    coordinator = AuditCoordinator(resolver=resolver)
    with coordinator.audit("novel", 1, 1, 0) as ctx:
        if ctx.collector:
            ctx.collector.record_stage("planning", inputs={"goal": "test"})
            ctx.collector.record_stage("draft", outputs={"text": "..."})

    trace = ctx.trace
    if trace:
        assert len(trace.stages) == 2
        assert trace.get_stage("planning") is not None
        assert trace.get_stage("draft") is not None


def test_coordinator_full_pipeline():
    resolver = MemoryPayloadResolver()
    coordinator = AuditCoordinator(resolver=resolver)

    with coordinator.audit("novel", 1, 1, 0) as ctx:
        if ctx.collector:
            plan_ref = PayloadRef("memory://planning/001")
            resolver.register(plan_ref, {"goal": "write", "outcome": "success", "must_events": ["A", "B", "C"]})
            plan_id = ctx.collector.record_reference("planning", plan_ref, "digest1", 100)
            ctx.collector.record_stage("planning", outputs={"plan_id": plan_id})

            prompt_ref = PayloadRef("memory://prompt/001")
            resolver.register(prompt_ref, {"goal": "write", "must_events": ["A"]})
            prompt_id = ctx.collector.record_reference("prompt_bundle", prompt_ref, "digest2", 200)
            ctx.collector.record_stage("prompt", inputs={"plan_id": plan_id}, outputs={"prompt_id": prompt_id})

            draft_ref = PayloadRef("memory://draft/001")
            resolver.register(draft_ref, {"goal": "write"})
            draft_id = ctx.collector.record_reference("draft", draft_ref, "digest3", 300)
            ctx.collector.record_stage("draft", inputs={"prompt_id": prompt_id}, outputs={"draft_id": draft_id})

    report = ctx.report
    assert report is not None
    assert report.execution_id == str(ctx.execution_id)
    assert report.summary["total_fields"] > 0
    assert report.summary["fields_with_loss"] > 0


def test_store_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = AuditReportStore(Path(tmpdir))
        from src.writing.audit.reporter import ComprehensiveReport
        from src.writing.audit.preservation import PreservationReport
        from src.writing.audit.attribution import AttributionReport
        from src.writing.audit.budget import BudgetReport
        from src.writing.audit.priority import PriorityReport

        pres_report = PreservationReport(
            execution_id="test-123",
            total_fields=1,
            fields={},
            lost_fields=[],
            preserved_fields=["goal"],
            partial_fields=[],
        )
        attr_report = AttributionReport(
            execution_id="test-123",
            total_fields_analyzed=1,
            fields_with_loss=0,
            fields_without_loss=1,
            attributions={},
            by_type={},
        )
        budget_report = BudgetReport(
            execution_id="test-123",
            metric="tokens",
            total_metric_value=100,
            stages=(),
            anomalies=(),
            stage_scores={},
        )
        priority_report = PriorityReport(
            execution_id="test-123",
            targets=(),
        )

        report = ComprehensiveReport(
            execution_id="test-123",
            preservation=pres_report,
            attribution=attr_report,
            budget=budget_report,
            priority=priority_report,
        )

        path = store.save(report, "novel-123")
        assert path.exists()

        entries = store.list(novel_id="novel-123")
        assert len(entries) == 1
        assert entries[0].execution_id == "test-123"

        loaded = store.load(entries[0])
        assert loaded is not None
        assert loaded.execution_id == "test-123"