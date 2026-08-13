# src/writing/snapshot/runtime/models.py
"""
Phase 11.2.2/11.2.4: RuntimeSnapshot — 扩展 PipelineSnapshot 增加 Runtime Capabilities
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.writing.snapshot.models import (
    PipelineSnapshot,
    SnapshotIdentity,
    SnapshotManifest,
    SnapshotMetadata,
)
from src.writing.artifact.planning import PlanningArtifact
from src.writing.ir.models import WriterIR
from src.writing.prompt.bundle import PromptBundle
from src.writing.render.trace import RenderTrace
from src.writing.coverage.models import CoverageReport
from src.capabilities.runtime import FrozenRuntimeCapabilityRegistry


@dataclass(frozen=True)
class RuntimeSnapshot:
    """
    Runtime 执行环境快照（扩展 PipelineSnapshot + Runtime Capabilities）。
    """
    identity: SnapshotIdentity
    manifest: SnapshotManifest
    metadata: SnapshotMetadata
    planning: PlanningArtifact
    writer_ir: WriterIR
    prompt_bundle: PromptBundle
    render_trace: RenderTrace
    draft: str
    coverage: CoverageReport
    timestamp: datetime
    runtime_capabilities: FrozenRuntimeCapabilityRegistry

    def __post_init__(self):
        # 强制类型检查：runtime_capabilities 必须是 FrozenRuntimeCapabilityRegistry
        if not isinstance(self.runtime_capabilities, FrozenRuntimeCapabilityRegistry):
            raise TypeError(
                f"runtime_capabilities must be FrozenRuntimeCapabilityRegistry, "
                f"got {type(self.runtime_capabilities).__name__}"
            )

    def to_pipeline_snapshot(self) -> PipelineSnapshot:
        """转换为 PipelineSnapshot（用于 Phase 10.1 兼容）。"""
        return PipelineSnapshot(
            identity=self.identity,
            manifest=self.manifest,
            metadata=self.metadata,
            planning=self.planning,
            writer_ir=self.writer_ir,
            prompt_bundle=self.prompt_bundle,
            render_trace=self.render_trace,
            draft=self.draft,
            coverage=self.coverage,
            timestamp=self.timestamp,
        )