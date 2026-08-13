# src/writing/snapshot/decoder.py

from dataclasses import fields, is_dataclass
from datetime import datetime, UTC
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar, Type
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
from src.writing.ir.models import WriterIR, IRDiagnostic
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


class SnapshotDecodeError(Exception):
    """Decoder 遇到无法处理的输入时抛出"""
    pass


T = TypeVar("T")


class SnapshotDecoder:
    """
    Builder 风格 Decoder，负责版本判断与对象重建。

    原则：
    - 缺失必需字段 → 抛出 SnapshotDecodeError
    - 可选字段缺失 → 使用默认值
    - 不支持的类型 → 抛出 SnapshotDecodeError
    - 不静默填充 identity 字段（如 timestamp）
    """

    def decode(self, data: Dict[str, Any]) -> PipelineSnapshot:
        """从原始 dict 重建 PipelineSnapshot"""
        timestamp = data.get("timestamp")
        if timestamp is None:
            raise SnapshotDecodeError("Missing required field: timestamp")

        return PipelineSnapshot(
            identity=self._decode_identity(data.get("identity", {})),
            manifest=self._decode_manifest(data.get("manifest", {})),
            metadata=self._decode_metadata(data.get("metadata", {})),
            planning=self._decode_planning(data.get("planning")),
            writer_ir=self._decode_writer_ir(data.get("writer_ir")),
            prompt_bundle=self._decode_prompt_bundle(data.get("prompt_bundle")),
            render_trace=self._decode_render_trace(data.get("render_trace")),
            draft=data.get("draft", ""),
            coverage=self._decode_coverage(data.get("coverage")),
            timestamp=self._decode_datetime(timestamp),
        )

    # ================================================================
    # Identity
    # ================================================================

    def _decode_identity(self, data: Dict[str, Any]) -> SnapshotIdentity:
        snapshot_id = data.get("snapshot_id")
        if not snapshot_id:
            raise SnapshotDecodeError("Missing required field: identity.snapshot_id")
        return SnapshotIdentity(
            snapshot_id=UUID(snapshot_id),
        )

    # ================================================================
    # Manifest
    # ================================================================

    def _decode_manifest(self, data: Dict[str, Any]) -> SnapshotManifest:
        return SnapshotManifest(
            schema_version=data.get("schema_version", "1.0"),
            format_version=data.get("format_version", "1.0"),
            serializer=data.get("serializer", "json"),
            compression=data.get("compression", "none"),
            checksum=data.get("checksum", "sha256"),
            created_by=data.get("created_by", "phase10.1"),
        )

    # ================================================================
    # Metadata
    # ================================================================

    def _decode_metadata(self, data: Dict[str, Any]) -> SnapshotMetadata:
        return SnapshotMetadata(
            runtime_version=data.get("runtime_version", ""),
            writer_version=data.get("writer_version", ""),
            llm_model=data.get("llm_model", ""),
            temperature=float(data.get("temperature", 0.0)),
            seed=data.get("seed"),
            git_commit=data.get("git_commit"),
            git_dirty=bool(data.get("git_dirty", False)),
            experiment_id=data.get("experiment_id"),
            python_version=data.get("python_version", ""),
            platform=data.get("platform", ""),
            os=data.get("os", ""),
            dependency_hash=data.get("dependency_hash", ""),
        )

    # ================================================================
    # Planning
    # ================================================================

    def _decode_planning(self, data: Optional[Dict[str, Any]]) -> PlanningArtifact:
        if data is None:
            # 返回默认空 PlanningArtifact
            return PlanningArtifact(
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

        core_data = data.get("core", {})
        if core_data is None:
            core_data = {}

        core = PlanningCore(
            scene_id=self._require_string(core_data, "scene_id"),
            scene_goal=self._require_string(core_data, "scene_goal"),
            must_events=self._require_list(core_data, "must_events"),
            world_state=self._decode_world_state(core_data.get("world_state", {})),
            conflicts=[
                self._decode_conflict(c) for c in core_data.get("conflicts", [])
            ],
            characters=[
                self._decode_character(c) for c in core_data.get("characters", [])
            ],
            emotion_arc=core_data.get("emotion_arc"),
        )

        return PlanningArtifact(
            core=core,
            extension=data.get("extension", {}),
            schema_version=data.get("schema_version", "1.0"),
        )

    def _decode_world_state(self, data: Dict[str, Any]) -> WorldStateArtifact:
        if data is None:
            data = {}
        return WorldStateArtifact(
            location=data.get("location", ""),
            time=data.get("time", ""),
            weather=data.get("weather", ""),
            realm=data.get("realm"),
        )

    def _decode_conflict(self, data: Dict[str, Any]) -> ConflictArtifact:
        if data is None:
            data = {}
        return ConflictArtifact(
            type=data.get("type", ""),
            description=data.get("description", ""),
            participants=data.get("participants", []),
            severity=data.get("severity"),
        )

    def _decode_character(self, data: Dict[str, Any]) -> CharacterArtifact:
        if data is None:
            data = {}
        return CharacterArtifact(
            id=data.get("id", ""),
            name=data.get("name", ""),
            role=data.get("role", ""),
            realm=data.get("realm"),
        )

    # ================================================================
    # Writer IR
    # ================================================================

    def _decode_writer_ir(self, data: Optional[Dict[str, Any]]) -> WriterIR:
        if data is None:
            return WriterIR(
                scene_goal="",
                facts={},
                preferences={},
                constraints=[],
                checklist=[],
                metadata={},
                schema_version="1.0",
            )
        return WriterIR(
            scene_goal=data.get("scene_goal", ""),
            facts=data.get("facts", {}),
            preferences=data.get("preferences", {}),
            constraints=data.get("constraints", []),
            checklist=data.get("checklist", []),
            metadata=data.get("metadata", {}),
            schema_version=data.get("schema_version", "1.0"),
        )

    # ================================================================
    # Prompt Bundle
    # ================================================================

    def _decode_prompt_bundle(self, data: Optional[Dict[str, Any]]) -> PromptBundle:
        if data is None:
            return PromptBundle(
                system_prompt="",
                sections=[],
                manifest=PromptManifest(
                    ir_schema="1.0",
                    renderer_versions={},
                    generation_profile="default",
                ),
                schema_version="1.0",
            )
        sections = [
            self._decode_prompt_section(s) for s in data.get("sections", [])
        ]
        manifest_data = data.get("manifest", {})
        if manifest_data is None:
            manifest_data = {}
        manifest = PromptManifest(
            ir_schema=manifest_data.get("ir_schema", "1.0"),
            renderer_versions=manifest_data.get("renderer_versions", {}),
            generation_profile=manifest_data.get("generation_profile", "default"),
            tokenizer=manifest_data.get("tokenizer"),
            language=manifest_data.get("language", "zh"),
        )
        return PromptBundle(
            system_prompt=data.get("system_prompt", ""),
            sections=sections,
            manifest=manifest,
            schema_version=data.get("schema_version", "1.0"),
        )

    def _decode_prompt_section(self, data: Dict[str, Any]) -> PromptSection:
        if data is None:
            data = {}
        return PromptSection(
            section_id=data.get("section_id", ""),
            renderer=data.get("renderer", ""),
            version=data.get("version", "1.0"),
            priority=data.get("priority", 0),
            content=data.get("content", ""),
            consumed_fields=data.get("consumed_fields", []),
        )

    # ================================================================
    # Render Trace
    # ================================================================

    def _decode_render_trace(self, data: Optional[Dict[str, Any]]) -> RenderTrace:
        if data is None:
            return RenderTrace(entries=[], total_elapsed_ms=0.0, schema_version="1.0")
        entries = [
            self._decode_render_entry(e) for e in data.get("entries", [])
        ]
        return RenderTrace(
            entries=entries,
            total_elapsed_ms=float(data.get("total_elapsed_ms", 0.0)),
            schema_version=data.get("schema_version", "1.0"),
        )

    def _decode_render_entry(self, data: Dict[str, Any]) -> RenderEntry:
        if data is None:
            data = {}
        status_str = data.get("status", "SUCCESS")
        try:
            status = RenderStatus[status_str]
        except KeyError:
            raise SnapshotDecodeError(f"Unknown RenderStatus: {status_str}")

        return RenderEntry(
            section_id=data.get("section_id", ""),
            renderer=data.get("renderer", ""),
            version=data.get("version", "1.0"),
            priority=data.get("priority", 0),
            status=status,
            chars=data.get("chars", 0),
            estimated_tokens=data.get("estimated_tokens", 0),
            elapsed_ms=float(data.get("elapsed_ms", 0.0)),
            consumed_fields=data.get("consumed_fields", []),
            error=data.get("error"),
        )

    # ================================================================
    # Coverage
    # ================================================================

    def _decode_coverage(self, data: Optional[Dict[str, Any]]) -> CoverageReport:
        if data is None:
            return CoverageReport(
                overall_score=0.0,
                structural_score=0.0,
                semantic_score=0.0,
                items=[],
                findings=[],
                grounding_breakdown={},
                schema_version="1.0",
            )
        items = [self._decode_coverage_item(i) for i in data.get("items", [])]
        findings = [self._decode_coverage_finding(f) for f in data.get("findings", [])]
        return CoverageReport(
            overall_score=float(data.get("overall_score", 0.0)),
            structural_score=float(data.get("structural_score", 0.0)),
            semantic_score=float(data.get("semantic_score", 0.0)),
            items=items,
            findings=findings,
            grounding_breakdown=data.get("grounding_breakdown", {}),
            schema_version=data.get("schema_version", "1.0"),
        )

    def _decode_coverage_item(self, data: Dict[str, Any]) -> CoverageItem:
        if data is None:
            data = {}
        status_str = data.get("status", "PASS")
        try:
            status = CoverageStatus[status_str]
        except KeyError:
            raise SnapshotDecodeError(f"Unknown CoverageStatus: {status_str}")

        evidence = [
            self._decode_evidence_ref(e) for e in data.get("evidence", [])
        ]
        return CoverageItem(
            item_id=data.get("item_id", ""),
            description=data.get("description", ""),
            status=status,
            score=float(data.get("score", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            evidence=evidence,
            reason=data.get("reason", ""),
        )

    def _decode_coverage_finding(self, data: Dict[str, Any]) -> CoverageFinding:
        if data is None:
            data = {}
        severity_str = data.get("severity", "INFO")
        try:
            severity = Severity[severity_str]
        except KeyError:
            raise SnapshotDecodeError(f"Unknown Severity: {severity_str}")

        category_str = data.get("category", "GROUNDING")
        try:
            category = CoverageCategory[category_str]
        except KeyError:
            raise SnapshotDecodeError(f"Unknown CoverageCategory: {category_str}")

        evidence_refs = [
            self._decode_evidence_ref(e) for e in data.get("evidence_refs", [])
        ]
        return CoverageFinding(
            severity=severity,
            category=category,
            target=data.get("target", ""),
            current=float(data.get("current", 0.0)),
            expected=float(data.get("expected", 0.0)),
            message=data.get("message", ""),
            evidence_refs=evidence_refs,
        )

    def _decode_evidence_ref(self, data: Dict[str, Any]) -> EvidenceReference:
        if data is None:
            data = {}
        return EvidenceReference(
            paragraph=data.get("paragraph", 0),
            sentence=data.get("sentence", 0),
            text=data.get("text", ""),
            start_char=data.get("start_char"),
            end_char=data.get("end_char"),
        )

    # ================================================================
    # Primitive Helpers
    # ================================================================

    def _decode_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                # 处理 ISO8601 格式，支持 "Z" 和 "+00:00"
                normalized = value.replace("Z", "+00:00")
                return datetime.fromisoformat(normalized)
            except ValueError as e:
                raise SnapshotDecodeError(f"Invalid timestamp format: {value}") from e
        raise SnapshotDecodeError(f"Unsupported timestamp type: {type(value).__name__}")

    def _require_string(self, data: Dict[str, Any], field: str) -> str:
        value = data.get(field)
        if not value:
            raise SnapshotDecodeError(f"Missing required field: {field}")
        if not isinstance(value, str):
            raise SnapshotDecodeError(
                f"Field {field} expected string, got {type(value).__name__}"
            )
        return value

    def _require_list(self, data: Dict[str, Any], field: str) -> List[Any]:
        value = data.get(field)
        if value is None:
            raise SnapshotDecodeError(f"Missing required field: {field}")
        if not isinstance(value, list):
            raise SnapshotDecodeError(
                f"Field {field} expected list, got {type(value).__name__}"
            )
        return value