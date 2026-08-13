# src/writing/audit/collector.py
"""
Phase 10.2.1: TraceCollector — 轻量级执行痕迹收集器
"""

from typing import Optional, Dict, Any, List, ContextManager, Set, Union
from datetime import datetime, timezone
from uuid import UUID, uuid4

from .registry import (
    StageRegistry,
    ArtifactTypeRegistry,
    create_default_stage_registry,
    create_default_artifact_type_registry,
)
from .payload_ref import PayloadRef
from .trace import Artifact, StageRecord, ExecutionTrace, SCHEMA_VERSION


class TraceCollector:
    """
    轻量级执行痕迹收集器。
    """

    def __init__(
        self,
        novel_id: str = "",
        volume: int = 0,
        chapter: int = 0,
        scene_idx: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        stage_registry: Optional[StageRegistry] = None,
        artifact_type_registry: Optional[ArtifactTypeRegistry] = None,
        strict_registry: bool = True,
        execution_id: Optional[UUID] = None,
    ):
        self._novel_id = novel_id
        self._volume = volume
        self._chapter = chapter
        self._scene_idx = scene_idx
        self._metadata = metadata or {}
        self._stage_registry = stage_registry or create_default_stage_registry()
        self._artifact_type_registry = artifact_type_registry or create_default_artifact_type_registry()
        self._strict_registry = strict_registry
        self._execution_id = execution_id or uuid4()
        self._start_time = datetime.now(timezone.utc)
        self._end_time: Optional[datetime] = None
        self._stages: List[StageRecord] = []
        self._artifacts: Dict[UUID, Artifact] = {}
        self._current_stage_builder: Optional[_StageBuilder] = None
        self._finished = False

    def _validate_stage(self, stage_id: str) -> None:
        if self._strict_registry and not self._stage_registry.is_valid(stage_id):
            raise ValueError(
                f"Unknown stage '{stage_id}'. "
                f"Registered stages: {self._stage_registry.list()}"
            )

    def _validate_artifact_type(self, artifact_type: str) -> None:
        if self._strict_registry and not self._artifact_type_registry.is_valid(artifact_type):
            raise ValueError(
                f"Unknown artifact type '{artifact_type}'. "
                f"Registered types: {self._artifact_type_registry.list()}"
            )

    def _validate_artifact_exists(self, artifact_id: UUID) -> None:
        if artifact_id not in self._artifacts:
            raise ValueError(
                f"Artifact '{artifact_id}' not found in trace. "
                "All input/output artifacts must be recorded before being referenced in a stage."
            )

    def record_artifact(self, artifact: Artifact) -> UUID:
        if self._finished:
            raise RuntimeError("Trace already finished")
        self._validate_artifact_type(artifact.artifact_type)
        if artifact.artifact_id in self._artifacts:
            raise ValueError(
                f"Artifact '{artifact.artifact_id}' already exists in this trace. "
                "Artifact IDs must be unique."
            )
        self._artifacts[artifact.artifact_id] = artifact
        return artifact.artifact_id

    def record_reference(
        self,
        artifact_type: str,
        payload_ref: PayloadRef,
        digest: str = "",
        size_bytes: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UUID:
        if self._finished:
            raise RuntimeError("Trace already finished")
        self._validate_artifact_type(artifact_type)
        artifact_id = uuid4()
        while artifact_id in self._artifacts:
            artifact_id = uuid4()
        artifact = Artifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            payload_ref=payload_ref,
            digest=digest,
            size_bytes=size_bytes,
            metadata=metadata or {},
        )
        self._artifacts[artifact.artifact_id] = artifact
        return artifact.artifact_id

    def record_stage(
        self,
        stage: str,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "TraceCollector":
        if self._finished:
            raise RuntimeError("Trace already finished")
        self._validate_stage(stage)

        if self._current_stage_builder is not None:
            self._end_stage(self._current_stage_builder)

        builder = _StageBuilder(self, stage)
        if metadata:
            builder.metrics(metadata)

        if inputs:
            for key, value in inputs.items():
                if isinstance(value, UUID):
                    builder.input(value)
        if outputs:
            for key, value in outputs.items():
                if isinstance(value, UUID):
                    builder.output(value)

        self._current_stage_builder = builder
        self._end_stage(builder)
        return self

    def stage(self, stage_id: str) -> ContextManager["_StageBuilder"]:
        self._validate_stage(stage_id)
        return _StageContext(self, stage_id)

    def _begin_stage(self, stage_id: str) -> "_StageBuilder":
        if self._finished:
            raise RuntimeError("Trace already finished")
        self._validate_stage(stage_id)
        builder = _StageBuilder(self, stage_id)
        self._current_stage_builder = builder
        return builder

    def _end_stage(self, builder: "_StageBuilder") -> None:
        if self._current_stage_builder is not builder:
            raise RuntimeError("Stage builder mismatch")

        for aid in builder._input_ids:
            self._validate_artifact_exists(aid)
        for aid in builder._output_ids:
            self._validate_artifact_exists(aid)

        record = builder.build()
        self._stages.append(record)
        self._current_stage_builder = None

    def finish(self) -> ExecutionTrace:
        if self._finished:
            raise RuntimeError("Trace already finished")
        if self._current_stage_builder is not None:
            raise RuntimeError("Cannot finish while a stage is still open")
        self._end_time = datetime.now(timezone.utc)
        self._finished = True
        return ExecutionTrace(
            execution_id=self._execution_id,
            schema_version=SCHEMA_VERSION,
            novel_id=self._novel_id,
            volume=self._volume,
            chapter=self._chapter,
            scene_idx=self._scene_idx,
            stages=tuple(self._stages),
            artifacts=dict(self._artifacts),
            start_time=self._start_time,
            end_time=self._end_time,
            metadata=self._metadata.copy(),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._finished:
            self.finish()

    def current_trace(self) -> Optional[ExecutionTrace]:
        if self._finished:
            return self.finish()
        return None


class _StageBuilder:
    def __init__(self, collector: TraceCollector, stage_id: str):
        self._collector = collector
        self._stage = stage_id
        self._stage_id = uuid4()
        self._start_time = datetime.now(timezone.utc)
        self._input_ids: List[UUID] = []
        self._output_ids: List[UUID] = []
        self._seen_inputs: Set[UUID] = set()
        self._seen_outputs: Set[UUID] = set()
        self._metrics: Dict[str, Any] = {}

    def input(self, artifact_id: UUID) -> "_StageBuilder":
        if artifact_id not in self._seen_inputs:
            self._seen_inputs.add(artifact_id)
            self._input_ids.append(artifact_id)
        return self

    def inputs(self, artifact_ids: List[UUID]) -> "_StageBuilder":
        for aid in artifact_ids:
            self.input(aid)
        return self

    def output(self, artifact_id: UUID) -> "_StageBuilder":
        if artifact_id not in self._seen_outputs:
            self._seen_outputs.add(artifact_id)
            self._output_ids.append(artifact_id)
        return self

    def outputs(self, artifact_ids: List[UUID]) -> "_StageBuilder":
        for aid in artifact_ids:
            self.output(aid)
        return self

    def metric(self, key: str, value: Any) -> "_StageBuilder":
        self._metrics[key] = value
        return self

    def metrics(self, metrics: Dict[str, Any]) -> "_StageBuilder":
        self._metrics.update(metrics)
        return self

    def build(self) -> StageRecord:
        return StageRecord(
            stage=self._stage,
            stage_id=self._stage_id,
            input_artifacts=tuple(self._input_ids),
            output_artifacts=tuple(self._output_ids),
            start_time=self._start_time,
            end_time=datetime.now(timezone.utc),
            metrics=self._metrics.copy(),
        )


class _StageContext:
    def __init__(self, collector: TraceCollector, stage_id: str):
        self._collector = collector
        self._stage = stage_id
        self._builder: Optional[_StageBuilder] = None

    def __enter__(self) -> _StageBuilder:
        self._builder = self._collector._begin_stage(self._stage)
        return self._builder

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._builder is not None:
            self._collector._end_stage(self._builder)
            self._builder = None