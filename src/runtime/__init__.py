"""
Runtime Core Package - Phase 7A/7B
"""

from .observation_ir import ObservationIR, SentenceSpan, PatternSpan, DocumentMetadata
from .observation_compiler import ObservationCompiler
from .validator import Validator, ComplianceReport, LayerComplianceResult, LayerComplianceEvidence
from .edit_compiler import EditCompiler, EditPlan, EditAction, EditOperation
from .patch_renderer import PatchRenderer, RenderedPatch

# Phase 7A 新增
from .snapshot import RuntimeSnapshot, RuntimeConfig, RuntimeMetrics
from .registry import SurfaceRegistry
from .builder import RuntimeBuilder
from .loader import PluginLoader
from .catalog import SurfaceCatalog

# Phase 7A 异常
from .exceptions import (
    RuntimeError,
    UnknownSurfaceError,
    DuplicateSurfaceError,
    SnapshotBuildError,
    RegistryFrozenError,
)

# Capability IDs
from src.capabilities import Matchers, Metrics, Repairs, Triggers


# ============================================================
# 向后兼容函数（Phase 6 接口，适配新 API）
# ============================================================

def _get_legacy_snapshot():
    """为向后兼容创建默认的 RuntimeSnapshot"""
    from src.surfaces.reasoning import ReasoningSurface
    registry = SurfaceRegistry((ReasoningSurface,))
    return RuntimeBuilder(registry).from_surfaces(registry, "reasoning")


def validate_draft(draft: str, layer_targets: dict = None) -> dict:
    """
    验证一段文本的合规性（生产环境入口）
    Phase 7B-2: 适配新 Validator API
    """
    compiler = ObservationCompiler()
    validator = Validator()
    snapshot = _get_legacy_snapshot()

    ir = compiler.compile(draft, snapshot)
    report = validator.validate(snapshot, ir)

    return {
        "compliance": report.overall_compliance,
        "layer_results": [
            {
                "layer": r.layer,
                "compliant": r.compliant,
                "target_level": r.target_level,
                "evidence_count": len(r.evidence_list)
            }
            for r in report.layer_results
        ],
        "ir_hash": ir.source_hash,
        "sentence_count": len(ir.sentences),
        "pattern_count": len(ir.patterns),
        "pattern_types": list(set(p.pattern_type for p in ir.patterns))
    }


def execute_with_diagnosis(draft: str, layer_targets: dict = None) -> dict:
    """
    执行完整诊断（生产环境入口）

    Args:
        draft: 待诊断的文本
        layer_targets: 层目标配置（已弃用，保留仅为向后兼容）

    Returns:
        dict: 包含诊断结果、合规报告、修订建议
    """
    if layer_targets is None:
        layer_targets = {
            "reasoning": "enhanced",
            "justification": "enhanced",
            "construction": "enhanced",
            "prediction": "enhanced",
        }

    compiler = ObservationCompiler()
    validator = Validator()
    edit_compiler = EditCompiler()
    renderer = PatchRenderer()
    snapshot = _get_legacy_snapshot()

    ir = compiler.compile(draft, snapshot)
    report = validator.validate(snapshot, ir)

    result = {
        "compliance": report.overall_compliance,
        "layer_results": [
            {
                "layer": r.layer,
                "compliant": r.compliant,
                "target_level": r.target_level,
                "evidence": [
                    {
                        "anchor_sentence_id": e.anchor_sentence_id,
                        "missing_pattern_types": e.missing_pattern_types
                    }
                    for e in r.evidence_list
                ]
            }
            for r in report.layer_results
        ],
        "ir_hash": ir.source_hash,
        "needs_revision": report.overall_compliance < 1.0
    }

    # 如果需要修订，生成修订计划
    if result["needs_revision"]:
        plan = edit_compiler.compile(ir, report, diagnosis_id="prod_diagnosis")
        if plan and plan.actions:
            rendered = renderer.render(plan, ir)
            result["revision_plan"] = {
                "actions": [a.to_dict() for a in plan.actions],
                "prompt_preview": rendered.full_prompt[:500] + "..." if len(rendered.full_prompt) > 500 else rendered.full_prompt
            }

    return result


__all__ = [
    # 核心类
    "ObservationCompiler",
    "ObservationIR",
    "SentenceSpan",
    "PatternSpan",
    "DocumentMetadata",
    "Validator",
    "ComplianceReport",
    "LayerComplianceResult",
    "LayerComplianceEvidence",
    "EditCompiler",
    "EditPlan",
    "EditAction",
    "EditOperation",
    "PatchRenderer",
    "RenderedPatch",
    # Phase 7A 新增
    "RuntimeSnapshot",
    "RuntimeConfig",
    "RuntimeMetrics",
    "SurfaceRegistry",
    "RuntimeBuilder",
    "PluginLoader",
    "SurfaceCatalog",
    # Phase 7A 异常
    "RuntimeError",
    "UnknownSurfaceError",
    "DuplicateSurfaceError",
    "SnapshotBuildError",
    "RegistryFrozenError",
    # Capability IDs
    "Matchers",
    "Metrics",
    "Repairs",
    "Triggers",
    # 生产兼容函数
    "validate_draft",
    "execute_with_diagnosis",
]