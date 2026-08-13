# src/writing/snapshot/models.py

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional
from uuid import UUID, uuid4

from src.writing.artifact.planning import PlanningArtifact
from src.writing.ir.models import WriterIR
from src.writing.prompt.bundle import PromptBundle
from src.writing.render.trace import RenderTrace
from src.writing.coverage.models import CoverageReport


@dataclass(frozen=True)
class SnapshotIdentity:
    snapshot_id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True)
class SnapshotManifest:
    schema_version: str = "1.0"
    format_version: str = "1.0"
    serializer: str = "json"
    compression: str = "none"
    checksum: str = "sha256"
    created_by: str = "phase10.1"


@dataclass(frozen=True)
class SnapshotMetadata:
    runtime_version: str = ""
    writer_version: str = ""
    llm_model: str = ""
    temperature: float = 0.0
    seed: Optional[int] = None
    git_commit: Optional[str] = None
    git_dirty: bool = False
    experiment_id: Optional[str] = None
    python_version: str = ""
    platform: str = ""
    os: str = ""
    dependency_hash: str = ""


@dataclass(frozen=True)
class PipelineSnapshot:
    identity: SnapshotIdentity
    manifest: SnapshotManifest
    metadata: SnapshotMetadata
    planning: PlanningArtifact
    writer_ir: WriterIR
    prompt_bundle: PromptBundle
    render_trace: RenderTrace
    draft: str
    coverage: CoverageReport
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))