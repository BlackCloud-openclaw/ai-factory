#!/usr/bin/env python
"""
重新生成 Golden 文件（使用真实的序列化路径）
必须与测试代码保持完全一致。
"""

import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib
from datetime import datetime, UTC
from uuid import UUID

from src.writing.snapshot import (
    PipelineSnapshot,
    SnapshotIdentity,
    SnapshotManifest,
    SnapshotMetadata,
    JsonSerializer,
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


def generate_golden():
    """生成 Golden 文件（固定 UUID 和时间戳）"""
    print("🔨 生成 Golden 文件（使用真实序列化路径）...")

    # 固定值
    fixed_uuid = UUID("00000000-0000-0000-0000-000000000001")
    fixed_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

    # 构造完整的 PipelineSnapshot
    identity = SnapshotIdentity(snapshot_id=fixed_uuid)
    manifest = SnapshotManifest(
        schema_version="1.0",
        format_version="1.0",
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

    # Planning Artifact
    planning = PlanningArtifact(
        core=PlanningCore(
            scene_id="test_scene",
            scene_goal="测试目标",
            must_events=["事件A", "事件B"],
            world_state=WorldStateArtifact(
                location="测试地点",
                time="清晨",
                weather="晴",
                realm="炼气"
            ),
            conflicts=[
                ConflictArtifact(
                    type="person",
                    description="角色冲突",
                    participants=["角色1", "角色2"],
                    severity="high"
                )
            ],
            characters=[
                CharacterArtifact(
                    id="char_1",
                    name="林逸",
                    role="protagonist",
                    realm="炼气三层"
                )
            ],
            emotion_arc={"begin": "平静", "end": "紧张"},
        ),
        extension={},
        schema_version="1.0",
    )

    # Writer IR
    writer_ir = WriterIR(
        scene_goal="测试目标",
        facts={"location": "测试地点", "time": "清晨"},
        preferences={"tone": "严肃"},
        constraints=["必须包含对话"],
        checklist=[{"id": "C1", "description": "检查点1"}],
        metadata={},
        schema_version="1.0",
    )

    # Prompt Bundle
    prompt_bundle = PromptBundle(
        system_prompt="你是一个测试助手",
        sections=[
            PromptSection(
                section_id="sec1",
                renderer="test_renderer",
                version="1.0",
                priority=10,
                content="测试内容",
                consumed_fields=["field1"]
            )
        ],
        manifest=PromptManifest(
            ir_schema="1.0",
            renderer_versions={"test_renderer": "1.0"},
            generation_profile="test",
            tokenizer="test",
            language="zh"
        ),
        schema_version="1.0",
    )

    # Render Trace
    render_trace = RenderTrace(
        entries=[
            RenderEntry(
                section_id="sec1",
                renderer="test_renderer",
                version="1.0",
                priority=10,
                status=RenderStatus.SUCCESS,
                chars=100,
                estimated_tokens=40,
                elapsed_ms=12.5,
                consumed_fields=["field1"],
                error=None
            )
        ],
        total_elapsed_ms=12.5,
        schema_version="1.0",
    )

    # Coverage Report
    coverage = CoverageReport(
        overall_score=0.9,
        structural_score=0.85,
        semantic_score=0.95,
        items=[
            CoverageItem(
                item_id="I1",
                description="覆盖项1",
                status=CoverageStatus.PASS,
                score=0.9,
                confidence=1.0,
                evidence=[EvidenceReference(paragraph=1, sentence=2, text="证据文本")],
                reason="覆盖良好"
            )
        ],
        findings=[
            CoverageFinding(
                severity=Severity.INFO,
                category=CoverageCategory.GROUNDING,
                target="目标字段",
                current=0.8,
                expected=1.0,
                message="轻微偏差",
                evidence_refs=[]
            )
        ],
        grounding_breakdown={"dim1": 0.9},
        schema_version="1.0",
    )

    # 构建最终 Snapshot
    snapshot = PipelineSnapshot(
        identity=identity,
        manifest=manifest,
        metadata=metadata,
        planning=planning,
        writer_ir=writer_ir,
        prompt_bundle=prompt_bundle,
        render_trace=render_trace,
        draft="林逸出手，长老让步。",
        coverage=coverage,
        timestamp=fixed_time,
    )

    # 使用真正的 JsonSerializer 进行序列化
    serializer = JsonSerializer()
    canonical_bytes = serializer.serialize(snapshot)

    # 写入文件
    golden_dir = Path("tests/fixtures/snapshots/v1.0")
    golden_dir.mkdir(parents=True, exist_ok=True)

    canonical_path = golden_dir / "canonical.json"
    canonical_path.write_bytes(canonical_bytes)

    sha256_path = golden_dir / "snapshot.sha256"
    digest = hashlib.sha256(canonical_bytes).hexdigest()
    sha256_path.write_bytes(digest.encode("ascii"))

    print(f"✅ Golden 文件生成: {canonical_path}")
    print(f"   SHA256: {digest}")

    # 验证：反序列化并检查 identity
    loaded_snapshot = serializer.deserialize(canonical_path.read_bytes())
    assert loaded_snapshot.identity.snapshot_id == fixed_uuid
    print("✅ 验证通过: 序列化/反序列化一致")


if __name__ == "__main__":
    generate_golden()