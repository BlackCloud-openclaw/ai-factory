# tests/fixtures/builders/snapshot_builder.py

from datetime import datetime, UTC
from uuid import UUID

from src.writing.snapshot.models import (
    PipelineSnapshot,
    SnapshotIdentity,
    SnapshotManifest,
    SnapshotMetadata,
)
from src.writing.artifact.planning import (
    PlanningArtifact,
    PlanningCore,
    WorldStateArtifact,
    ConflictArtifact,
    CharacterArtifact,
)
from src.writing.ir.models import WriterIR
from src.writing.prompt.bundle import PromptBundle, PromptSection, PromptManifest
from src.writing.render.trace import RenderTrace, RenderEntry, RenderStatus
from src.writing.coverage.models import (
    CoverageReport,
    CoverageItem,
    CoverageFinding,
    CoverageStatus,
    CoverageCategory,
    EvidenceReference,
)
from src.writing.common.severity import Severity


# ✅ 固定 UUID 和时间戳，确保 Golden 稳定
FIXED_UUID = UUID("00000000-0000-0000-0000-000000000001")
FIXED_TIMESTAMP = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def build_sample_snapshot() -> PipelineSnapshot:
    """构建一个确定性的 PipelineSnapshot，用于 Golden 测试"""
    identity = SnapshotIdentity(snapshot_id=FIXED_UUID)
    manifest = SnapshotManifest(
        schema_version="1.0",
        format_version="1.0",
        serializer="json",
        compression="none",
        checksum="sha256",
        created_by="phase10.1",
    )
    metadata = SnapshotMetadata(
        runtime_version="1.0",
        writer_version="1.0",
        llm_model="test",
        temperature=0.7,
        seed=42,
        git_commit="abc123",
        experiment_id="exp_001",
        python_version="3.11.0",
        platform="test",
        os="test",
        dependency_hash="test",
    )

    world_state = WorldStateArtifact(
        location="灵脉核心区",
        time="正午",
        weather="晴",
        realm="筑基",
    )
    conflict = ConflictArtifact(
        type="权争",
        description="灵脉分配争执",
        participants=["林逸", "长老"],
        severity="高",
    )
    character = CharacterArtifact(
        id="linyi",
        name="林逸",
        role="主角",
        realm="筑基三层",
    )
    core = PlanningCore(
        scene_id="scene_1",
        scene_goal="夺取灵脉控制权",
        must_events=["林逸出手", "长老让步"],
        world_state=world_state,
        conflicts=[conflict],
        characters=[character],
        emotion_arc={"begin": "好奇", "end": "坚定"},
    )
    planning = PlanningArtifact(
        core=core,
        extension={"builtin.voice": {"style": "短句"}},
        schema_version="1.0",
    )

    writer_ir = WriterIR(
        scene_goal="夺取灵脉控制权",
        facts={"key": "value"},
        preferences={"voice": "short"},
        constraints=["不得杀生"],
        checklist=[{"item": "提及玉佩"}],
        metadata={"source": "planning"},
        schema_version="1.0",
    )

    prompt_section = PromptSection(
        section_id="sec_001",
        renderer="EmotionRenderer",
        version="1.0",
        priority=20,
        content="请使用短句描写情绪",
        consumed_fields=["emotion_arc"],
    )
    prompt_manifest = PromptManifest(
        ir_schema="1.0",
        renderer_versions={"EmotionRenderer": "1.0"},
        generation_profile="quality",
        tokenizer="default",
        language="zh",
    )
    prompt_bundle = PromptBundle(
        system_prompt="你是小说编辑",
        sections=[prompt_section],
        manifest=prompt_manifest,
        schema_version="1.0",
    )

    render_entry = RenderEntry(
        section_id="sec_001",
        renderer="EmotionRenderer",
        version="1.0",
        priority=20,
        status=RenderStatus.SUCCESS,
        chars=120,
        estimated_tokens=48,
        elapsed_ms=15.0,
        consumed_fields=["emotion_arc"],
        error=None,
    )
    render_trace = RenderTrace(
        entries=[render_entry],
        total_elapsed_ms=15.0,
        schema_version="1.0",
    )

    evidence = EvidenceReference(
        paragraph=1,
        sentence=2,
        text="玉佩发热",
        start_char=10,
        end_char=20,
    )
    coverage_item = CoverageItem(
        item_id="check_001",
        description="提及玉佩",
        status=CoverageStatus.PASS,
        score=1.0,
        confidence=0.95,
        evidence=[evidence],
        reason="找到了",
    )
    coverage_finding = CoverageFinding(
        severity=Severity.INFO,
        category=CoverageCategory.GROUNDING,
        target="world_state",
        current=0.8,
        expected=0.9,
        message="世界状态覆盖良好",
        evidence_refs=[evidence],
    )
    coverage = CoverageReport(
        overall_score=0.85,
        structural_score=0.90,
        semantic_score=0.80,
        items=[coverage_item],
        findings=[coverage_finding],
        grounding_breakdown={"entity": 0.9, "state": 0.8},
        schema_version="1.0",
    )

    return PipelineSnapshot(
        identity=identity,
        manifest=manifest,
        metadata=metadata,
        planning=planning,
        writer_ir=writer_ir,
        prompt_bundle=prompt_bundle,
        render_trace=render_trace,
        draft="林逸出手，长老让步。",
        coverage=coverage,
        timestamp=FIXED_TIMESTAMP,
    )