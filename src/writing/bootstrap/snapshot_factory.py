# src/writing/bootstrap/snapshot_factory.py
"""
Phase 11.2.4: Snapshot Factory — 构造 RuntimeSnapshot
"""

from datetime import datetime

from src.writing.snapshot.runtime.models import RuntimeSnapshot
from src.writing.snapshot.runtime.id import SnapshotId
from src.writing.snapshot.models import SnapshotManifest, SnapshotMetadata
from src.writing.artifact.planning import PlanningArtifact, PlanningCore, WorldStateArtifact
from src.writing.ir.models import WriterIR
from src.writing.prompt.bundle import PromptBundle, PromptManifest
from src.writing.render.trace import RenderTrace
from src.writing.coverage.models import CoverageReport
from .composition_root import WriterRuntime


def build_runtime_snapshot(
    writer_runtime: WriterRuntime,
    novel_id: str = "",
    volume: int = 1,
    chapter: int = 1,
    scene_idx: int = 0,
) -> RuntimeSnapshot:
    """
    使用给定的 WriterRuntime 构造 RuntimeSnapshot。

    Args:
        writer_runtime: 已构建的 WriterRuntime
        novel_id: 小说 ID
        volume: 卷号
        chapter: 章号
        scene_idx: 场景索引

    Returns:
        包含 runtime_capabilities 的 RuntimeSnapshot
    """
    # 使用默认空值构造必要的 Artifact（实际将由 Writer 填充）
    identity = SnapshotId.new()
    manifest = SnapshotManifest(schema_version="1.0", format_version="1.0", serializer="json")
    metadata = SnapshotMetadata(
        runtime_version="1.0",
        writer_version="1.0",
        llm_model="",
        temperature=0.0,
    )

    planning = PlanningArtifact(
        core=PlanningCore(
            scene_id="",
            scene_goal="",
            must_events=[],
            world_state=WorldStateArtifact(location="", time="", weather=""),
            conflicts=[],
            characters=[],
        ),
        extension={},
        schema_version="1.0",
    )
    writer_ir = WriterIR(
        scene_goal="",
        facts={},
        preferences={},
        constraints=[],
        checklist=[],
        metadata={},
    )
    prompt_bundle = PromptBundle(
        system_prompt="",
        sections=[],
        manifest=PromptManifest(
            ir_schema="1.0",
            renderer_versions={},
            generation_profile="default",
        ),
    )
    render_trace = RenderTrace(entries=[], total_elapsed_ms=0.0)
    coverage = CoverageReport(
        overall_score=0.0,
        structural_score=0.0,
        semantic_score=0.0,
        items=[],
        findings=[],
        grounding_breakdown={},
    )

    return RuntimeSnapshot(
        identity=identity,
        manifest=manifest,
        metadata=metadata,
        planning=planning,
        writer_ir=writer_ir,
        prompt_bundle=prompt_bundle,
        render_trace=render_trace,
        draft="",
        coverage=coverage,
        timestamp=datetime.now(),
        runtime_capabilities=writer_runtime.runtime_capabilities,
    )