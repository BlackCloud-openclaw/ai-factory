# tests/unit/writing/audit/test_trace.py

import pytest
from types import MappingProxyType
from uuid import UUID, uuid4

from src.writing.audit import (
    TraceCollector,
    ExecutionTrace,
    StageRecord,
    Artifact,
    PayloadRef,
    StageRegistry,
    ArtifactTypeRegistry,
    StageDefinition,
    ArtifactTypeDefinition,
)


class TestExecutionTrace:

    def test_basic_flow(self):
        with TraceCollector("novel_1", 1, 1, 0) as collector:
            plan_id = collector.record_reference(
                "planning",
                PayloadRef("memory://planning/001"),
                digest="abc123",
                size_bytes=100,
            )
            with collector.stage("planning") as stage:
                stage.output(plan_id).metric("tokens", 100)

            prompt_id = collector.record_reference(
                "prompt_bundle",
                PayloadRef("memory://prompt/001"),
                digest="def456",
                size_bytes=1000,
            )
            with collector.stage("prompt") as stage:
                stage.input(plan_id).output(prompt_id).metric("tokens", 200)

            trace = collector.finish()

        assert len(trace.stages) == 2
        assert trace.get_stage("planning") is not None
        assert trace.get_stage("prompt") is not None
        assert trace.get_stage("unknown") is None

    def test_find_stages(self):
        with TraceCollector("novel", 1, 1, 0) as collector:
            a1 = collector.record_reference("planning", PayloadRef("mem://1"), "d1", 10)
            a2 = collector.record_reference("draft", PayloadRef("mem://2"), "d2", 20)
            a3 = collector.record_reference("draft", PayloadRef("mem://3"), "d3", 30)

            with collector.stage("planning") as s:
                s.output(a1)
            with collector.stage("draft") as s:
                s.input(a1).output(a2)
            with collector.stage("draft") as s:
                s.input(a2).output(a3)

            trace = collector.finish()

        drafts = trace.find_stages("draft")
        assert len(drafts) == 2
        assert drafts[0].stage == "draft"
        assert drafts[1].stage == "draft"

    def test_parents_children(self):
        with TraceCollector("novel", 1, 1, 0) as collector:
            a1 = collector.record_reference("planning", PayloadRef("mem://1"), "d1", 10)
            a2 = collector.record_reference("prompt_bundle", PayloadRef("mem://2"), "d2", 20)

            with collector.stage("planning") as s:
                s.output(a1)
            with collector.stage("prompt") as s:
                s.input(a1).output(a2)

            trace = collector.finish()

        parents = trace.parents(a2)
        assert len(parents) == 1
        assert parents[0].artifact_id == a1

        children = trace.children(a1)
        assert len(children) == 1
        assert children[0].artifact_id == a2

    def test_upstream_downstream(self):
        with TraceCollector("novel", 1, 1, 0) as collector:
            a1 = collector.record_reference("planning", PayloadRef("mem://1"), "d1", 10)
            a2 = collector.record_reference("prompt_bundle", PayloadRef("mem://2"), "d2", 20)
            a3 = collector.record_reference("draft", PayloadRef("mem://3"), "d3", 30)

            with collector.stage("planning") as s:
                s.output(a1)
            with collector.stage("prompt") as s:
                s.input(a1).output(a2)
            with collector.stage("draft") as s:
                s.input(a2).output(a3)

            trace = collector.finish()
            planning = trace.get_stage("planning")
            prompt = trace.get_stage("prompt")
            assert planning is not None
            assert prompt is not None

            # upstream 应该只包含 planning
            upstream = trace.upstream(prompt.stage_id)
            assert len(upstream) == 1
            assert upstream[0].stage == "planning"

            # downstream 应该包含 prompt 和 draft（因为 planning → prompt → draft）
            downstream = trace.downstream(planning.stage_id)
            assert len(downstream) == 2
            downstream_stages = [s.stage for s in downstream]
            assert "prompt" in downstream_stages
            assert "draft" in downstream_stages

    def test_immutable(self):
        with TraceCollector("novel", 1, 1, 0) as collector:
            aid = collector.record_reference("planning", PayloadRef("mem://1"), "d", 10)
            with collector.stage("planning") as s:
                s.output(aid)
            trace = collector.finish()

        assert isinstance(trace.artifacts, MappingProxyType)
        assert isinstance(trace.metadata, MappingProxyType)

        with pytest.raises(TypeError):
            trace.artifacts[aid] = None  # type: ignore

        with pytest.raises(AttributeError):
            trace.stages.append(None)  # type: ignore

    def test_stage_builder_deduplication(self):
        with TraceCollector("novel", 1, 1, 0) as collector:
            aid = collector.record_reference("planning", PayloadRef("mem://1"), "d", 10)
            with collector.stage("planning") as s:
                s.input(aid)
                s.input(aid)
                s.output(aid)
                s.output(aid)

            trace = collector.finish()
            stage = trace.get_stage("planning")
            assert stage is not None
            assert len(stage.input_artifacts) == 1
            assert len(stage.output_artifacts) == 1

    def test_registry_duplicate_registration(self):
        registry = StageRegistry()
        registry.register(StageDefinition("planning"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(StageDefinition("planning"))

        artifact_registry = ArtifactTypeRegistry()
        artifact_registry.register(ArtifactTypeDefinition("planning"))
        with pytest.raises(ValueError, match="already registered"):
            artifact_registry.register(ArtifactTypeDefinition("planning"))

    def test_collector_validation(self):
        with pytest.raises(ValueError, match="Unknown stage 'unknown_stage'"):
            with TraceCollector("novel", 1, 1, 0) as collector:
                collector.stage("unknown_stage")

        with pytest.raises(ValueError, match="Unknown artifact type 'unknown_type'"):
            with TraceCollector("novel", 1, 1, 0) as collector:
                collector.record_reference("unknown_type", PayloadRef("mem://1"), "d", 10)

        collector = TraceCollector("novel", 1, 1, 0, strict_registry=False)
        with collector:
            aid = collector.record_reference("unknown_type", PayloadRef("mem://1"), "d", 10)
            with collector.stage("unknown_stage") as s:
                s.output(aid)
            trace = collector.finish()
            assert trace.get_stage("unknown_stage") is not None

    def test_record_artifact_duplicate_id(self):
        with TraceCollector("novel", 1, 1, 0) as collector:
            aid = uuid4()
            artifact = Artifact(artifact_id=aid, artifact_type="planning")
            collector.record_artifact(artifact)
            with pytest.raises(ValueError, match="already exists"):
                collector.record_artifact(artifact)

    def test_duplicate_producer_detection(self):
        collector = TraceCollector("novel", 1, 1, 0)
        aid = collector.record_reference("planning", PayloadRef("mem://1"), "d1", 10)

        with collector.stage("planning") as s:
            s.output(aid)

        with collector.stage("draft") as s:
            s.output(aid)

        with pytest.raises(ValueError, match="produced by multiple stages"):
            collector.finish()

    def test_get_stage_by_uuid_optimized(self):
        with TraceCollector("novel", 1, 1, 0) as collector:
            aid = collector.record_reference("planning", PayloadRef("mem://1"), "d1", 10)
            with collector.stage("planning") as s:
                s.output(aid)
            trace = collector.finish()

        stage = trace.get_stage("planning")
        assert stage is not None
        assert trace._stage_index is not None
        assert trace.get_stage_by_uuid(stage.stage_id) is not None

    def test_unknown_artifact_reference(self):
        collector = TraceCollector("novel", 1, 1, 0)
        with pytest.raises(ValueError, match="Artifact .* not found"):
            with collector.stage("planning") as s:
                s.output(uuid4())

    def test_execution_id_persists(self):
        with TraceCollector("novel", 1, 1, 0) as collector:
            eid = collector._execution_id
            trace = collector.finish()
        assert trace.execution_id == eid

    def test_metrics_semantic(self):
        with TraceCollector("novel", 1, 1, 0) as collector:
            aid = collector.record_reference("planning", PayloadRef("mem://1"), "d", 10)
            with collector.stage("planning") as s:
                s.output(aid)
                s.metric("tokens", 100)
                s.metric("tokens", 200)
                s.metric("cost", 0.01)
            trace = collector.finish()

        stage = trace.get_stage("planning")
        assert stage is not None
        assert stage.metrics.get("tokens") == 200
        assert stage.metrics.get("cost") == 0.01