"""
Validator - 基于 RuntimeSnapshot 的合规验证
Phase 7B-2: 完全由 Snapshot 驱动，无 layer_targets 依赖
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any
from enum import Enum

from src.runtime.observation_ir import ObservationIR, SentenceSpan, PatternSpan
from src.runtime.snapshot import RuntimeSnapshot
from src.surfaces.definition import LayerRule, MetricDefinition, SurfaceDefinition


# ============================================================
# 1. 合规报告数据结构
# ============================================================

@dataclass
class LayerComplianceEvidence:
    """
    合规证据：全部使用 ID 引用，不复制文本。
    """
    anchor_sentence_id: str
    present_patterns: List[str]
    missing_pattern_types: List[str]
    conflicting_pattern_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchor_sentence_id": self.anchor_sentence_id,
            "present_patterns": self.present_patterns,
            "missing_pattern_types": self.missing_pattern_types,
            "conflicting_pattern_ids": self.conflicting_pattern_ids
        }


@dataclass
class LayerComplianceResult:
    """单层合规结果"""
    layer: str
    target_level: str
    compliant: bool
    evidence_list: List[LayerComplianceEvidence]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "target_level": self.target_level,
            "compliant": self.compliant,
            "evidence": [e.to_dict() for e in self.evidence_list]
        }


@dataclass
class ComplianceReport:
    """Validator 的唯一输出"""
    source_hash: str
    layer_results: List[LayerComplianceResult]
    overall_compliance: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_hash": self.source_hash,
            "layer_results": [r.to_dict() for r in self.layer_results],
            "overall_compliance": self.overall_compliance
        }


# ============================================================
# 2. Validator（纯规则引擎，由 Snapshot 驱动）
# ============================================================

class Validator:
    """
    Validator 是一个纯规则引擎，由 RuntimeSnapshot 驱动。
    
    Phase 7B-2 变化：
    - 输入从 (ir, layer_targets) 改为 (snapshot, ir)
    - 从 snapshot.surfaces 中获取所有 Layer 规则
    - 无任何 Surface 特判代码
    - 无 layer_targets 依赖
    """

    def validate(self, snapshot: RuntimeSnapshot, ir: ObservationIR) -> ComplianceReport:
        """
        验证 ObservationIR 是否符合 Snapshot 中所有 Surface 的 Layer 规则
        
        Args:
            snapshot: RuntimeSnapshot（包含所有 Surface 和配置）
            ir: ObservationIR（待验证的观察结果）
        
        Returns:
            ComplianceReport: 合规报告
        """
        layer_results: List[LayerComplianceResult] = []

        # 遍历所有 Surface 的 Layer 规则
        for surface in snapshot.surfaces:
            for rule in surface.validation.layer_rules:
                result = self._evaluate_layer(rule, ir, surface)
                layer_results.append(result)

        # 计算整体合规率
        total = len(layer_results)
        compliant_count = sum(1 for r in layer_results if r.compliant)
        overall = compliant_count / total if total > 0 else 1.0

        return ComplianceReport(
            source_hash=ir.source_hash,
            layer_results=layer_results,
            overall_compliance=overall,
        )

    def _evaluate_layer(self, rule: LayerRule, ir: ObservationIR, surface: SurfaceDefinition) -> LayerComplianceResult:
        """
        评估单层规则
        
        流程：
        1. 检查 required_types 是否全部存在
        2. 检查所有 metrics 是否通过
        """
        # 1. 检查 required_types
        required_present = True
        missing_types: List[str] = []
        present_patterns: List[str] = []

        for req_type in rule.required_types:
            found_patterns = [p for p in ir.patterns if p.pattern_type == req_type]
            if found_patterns:
                present_patterns.extend([p.id for p in found_patterns])
            else:
                required_present = False
                missing_types.append(req_type)

        # 2. 检查 metrics
        metrics_passed = True
        for metric in rule.metrics:
            if not self._evaluate_metric(metric, ir, snapshot=None):
                metrics_passed = False
                break

        compliant = required_present and metrics_passed

        # 3. 生成证据
        evidence_list: List[LayerComplianceEvidence] = []
        if not compliant:
            anchor = self._find_anchor_sentence(ir, rule.required_types)
            evidence_list.append(
                LayerComplianceEvidence(
                    anchor_sentence_id=anchor.id if anchor else "",
                    present_patterns=present_patterns,
                    missing_pattern_types=missing_types,
                )
            )

        return LayerComplianceResult(
            layer=rule.layer,
            target_level="enhanced",  # Phase 7: 由 Surface 自身定义
            compliant=compliant,
            evidence_list=evidence_list,
        )

    def _evaluate_metric(self, metric: MetricDefinition, ir: ObservationIR, snapshot: Optional[RuntimeSnapshot]) -> bool:
        """
        评估度量指标
        
        支持的 Metric：
        - dialogue_exists: 是否存在 dialogue_marker
        - 未来可扩展更多
        """
        if metric.name == "dialogue_exists":
            return any(p.pattern_type == "dialogue_marker" for p in ir.patterns)
        
        # 未知 Metric 默认通过
        return True

    def _find_anchor_sentence(self, ir: ObservationIR, required_types: List[str]) -> Optional[SentenceSpan]:
        """找到包含 required_types 中任意类型的句子作为锚点"""
        if not ir.sentences:
            return None

        for sent in ir.sentences:
            sent_patterns = [p for p in ir.patterns if p.sentence_id == sent.id]
            for req_type in required_types:
                if any(p.pattern_type == req_type for p in sent_patterns):
                    return sent

        # 如果没有匹配，返回第一个句子
        return ir.sentences[0]


# ============================================================
# 3. 快速测试
# ============================================================

if __name__ == "__main__":
    from src.runtime.observation_compiler import ObservationCompiler
    from src.runtime.loader import PluginLoader
    from src.runtime.registry import SurfaceRegistry
    from src.runtime.builder import RuntimeBuilder
    from src.runtime.snapshot import RuntimeConfig

    # 加载 Surface
    surfaces = PluginLoader.load_from_manifest()
    registry = SurfaceRegistry(surfaces)
    builder = RuntimeBuilder(registry)
    config = RuntimeConfig(enabled_surfaces=("reasoning", "dialogue"))
    snapshot = builder.with_config(config).build()

    # 编译和验证
    compiler = ObservationCompiler()
    validator = Validator()

    text = '林逸说：「这是对话。」'
    ir = compiler.compile(text, snapshot)
    report = validator.validate(snapshot, ir)

    print("=" * 60)
    print("Validator Test (Snapshot-driven)")
    print("=" * 60)
    print(f"Overall Compliance: {report.overall_compliance:.2f}")
    for r in report.layer_results:
        print(f"  {r.layer}: {'✅' if r.compliant else '❌'}")
        if not r.compliant:
            for e in r.evidence_list:
                print(f"    Missing: {e.missing_pattern_types}")