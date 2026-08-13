# tests/writing/snapshot/test_snapshot_core.py
"""
Snapshot 核心功能测试（直接从子模块导入，绕过 __init__.py）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# 直接从子模块导入
from src.writing.snapshot.models import (
    PipelineSnapshot,
    SnapshotIdentity,
    SnapshotManifest,
    SnapshotMetadata,
)
from src.writing.snapshot.serializer import JsonSerializer
from src.writing.snapshot.writer import SnapshotWriter
from src.writing.snapshot.loader import SnapshotLoader
from src.writing.snapshot.encoder import SnapshotEncoder
from src.writing.snapshot.decoder import SnapshotDecoder
from src.writing.snapshot.canonical_json import dumps

from src.writing.artifact.planning import PlanningArtifact, PlanningCore, WorldStateArtifact
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

import pytest
from datetime import datetime, UTC
from uuid import UUID


def create_test_snapshot():
    """创建测试用 Snapshot"""
    identity = SnapshotIdentity(snapshot_id=UUID("00000000-0000-0000-0000-000000000001"))
    manifest = SnapshotManifest()
    metadata = SnapshotMetadata(
        runtime_version="1.0",
        writer_version="1.0",
        llm_model="test",
        temperature=0.7,
        seed=42,
    )
    # ... 其他字段使用默认值
    return PipelineSnapshot(
        identity=identity,
        manifest=manifest,
        metadata=metadata,
        planning=PlanningArtifact(
            core=PlanningCore(
                scene_id="test",
                scene_goal="test",
                must_events=[],
                world_state=WorldStateArtifact(location="test", time="test", weather="test"),
                conflicts=[],
                characters=[],
            ),
            extension={},
        ),
        writer_ir=WriterIR(
            scene_goal="test",
            facts={},
            preferences={},
            constraints=[],
            checklist=[],
            metadata={},
        ),
        prompt_bundle=PromptBundle(
            system_prompt="",
            sections=[],
            manifest=PromptManifest(
                ir_schema="1.0",
                renderer_versions={},
                generation_profile="default",
            ),
        ),
        render_trace=RenderTrace(entries=[], total_elapsed_ms=0),
        draft="",
        coverage=CoverageReport(
            overall_score=0,
            structural_score=0,
            semantic_score=0,
            items=[],
            findings=[],
            grounding_breakdown={},
        ),
        timestamp=datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC),
    )


def test_roundtrip():
    snapshot = create_test_snapshot()
    serializer = JsonSerializer()
    data = serializer.serialize(snapshot)
    restored = serializer.deserialize(data)
    assert snapshot == restored


def test_golden_match():
    import hashlib
    golden_path = Path("tests/fixtures/snapshots/v1.0/canonical.json")
    if golden_path.exists():
        data = golden_path.read_bytes()
        sha256_path = golden_path.parent / "snapshot.sha256"
        expected = sha256_path.read_text().strip()
        actual = hashlib.sha256(data).hexdigest()
        assert actual == expected