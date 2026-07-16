# src/runtime/revision_analyzer.py
"""
Revision Analyzer - 分析修订为什么失败
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List

from src.runtime.revision_trace import RevisionTrace
from src.runtime.validator import ComplianceReport


class RevisionFailure(Enum):
    """修订失败类型"""
    # 执行层失败
    NO_CHANGE = "no_change"                       # LLM 返回的文本与原文完全相同
    MINOR_CHANGE = "minor_change"                 # 有变化但未触及目标层
    LLM_IGNORED_PROMPT = "llm_ignored_prompt"     # LLM 未遵循修订指令
    PATCH_DESTROYED_OTHER_LAYER = "patch_destroyed_other_layer"  # 修复一层却破坏了另一层
    
    # 验证层失败
    VALIDATOR_UNCHANGED = "validator_unchanged"   # 修订有效但 Validator 未识别
    VALIDATOR_REGRESSION = "validator_regression" # 修订导致合规率下降
    
    # 成功
    PATCH_SUCCESS = "patch_success"


@dataclass
class RevisionDiagnosis:
    """修订诊断结果"""
    failure_type: RevisionFailure
    confidence: float
    evidence: List[str]
    recommendation: str


class RevisionAnalyzer:
    """
    修订分析器 - 分析修订失败的原因
    """
    
    def analyze(self, trace: RevisionTrace) -> RevisionDiagnosis:
        """分析修订执行结果"""
        
        # 1. 检查是否有任何变化
        if not trace.has_change:
            return RevisionDiagnosis(
                failure_type=RevisionFailure.NO_CHANGE,
                confidence=0.98,
                evidence=["LLM 返回的文本与原文完全相同"],
                recommendation="检查 Patch Prompt 是否清晰，LLM 是否有权限修改文本"
            )
        
        # 2. 检查合规率是否改善
        if trace.compliance_improved:
            return RevisionDiagnosis(
                failure_type=RevisionFailure.PATCH_SUCCESS,
                confidence=0.90,
                evidence=[f"合规率从 {trace.validator_before.compliance_rate:.2f} 提升到 {trace.validator_after.compliance_rate:.2f}"],
                recommendation="修订有效"
            )
        
        # 3. 检查是否有退化
        if trace.compliance_degraded:
            return RevisionDiagnosis(
                failure_type=RevisionFailure.VALIDATOR_REGRESSION,
                confidence=0.80,
                evidence=[
                    f"合规率从 {trace.validator_before.compliance_rate:.2f} 下降到 {trace.validator_after.compliance_rate:.2f}",
                    "修订可能导致其他层被破坏"
                ],
                recommendation="检查 Patch 是否越界修改了其他层，考虑缩小修订范围"
            )
        
        # 4. 检查目标层是否被触及
        target_layers = set(trace.target_layers)
        before_compliant = self._get_compliant_layers(trace.validator_before)
        after_compliant = self._get_compliant_layers(trace.validator_after)
        
        # 检查目标层是否有变化
        layers_changed = False
        for layer in target_layers:
            if (layer in before_compliant) != (layer in after_compliant):
                layers_changed = True
                break
        
        if not layers_changed:
            return RevisionDiagnosis(
                failure_type=RevisionFailure.MINOR_CHANGE,
                confidence=0.75,
                evidence=[
                    f"有变化但目标层 {list(target_layers)} 的合规状态未改变",
                    f"原文长度: {len(trace.original_text)}, 修订后长度: {len(trace.revised_text)}",
                    trace.diff_summary
                ],
                recommendation="Patch Prompt 可能没有明确指定目标层，或 LLM 未优先处理目标层"
            )
        
        # 5. 检查是否修复了一层但破坏了另一层
        if after_compliant and before_compliant:
            # 检查是否有新的层变得不合规
            lost_layers = before_compliant - after_compliant
            if lost_layers:
                return RevisionDiagnosis(
                    failure_type=RevisionFailure.PATCH_DESTROYED_OTHER_LAYER,
                    confidence=0.80,
                    evidence=[
                        f"修复了目标层，但破坏了其他层: {list(lost_layers)}",
                        f"修订前合规层: {list(before_compliant)}",
                        f"修订后合规层: {list(after_compliant)}"
                    ],
                    recommendation="限制修订范围，使用 preserve_layers 保护其他层"
                )
        
        # 6. 默认：Validator 可能没有识别出有效的 State 引用
        return RevisionDiagnosis(
            failure_type=RevisionFailure.VALIDATOR_UNCHANGED,
            confidence=0.60,
            evidence=[
                f"修订有变化但合规率未变 ({trace.validator_before.compliance_rate:.2f} → {trace.validator_after.compliance_rate:.2f})",
                "可能 Validator 的规则过于严格，或 State 引用方式未被识别"
            ],
            recommendation="检查 Validator 规则是否过于严格，或 State 引用是否显式绑定了推理标记"
        )
    
    def _get_compliant_layers(self, report: ComplianceReport) -> set:
        """获取合规的层名称"""
        layers = set()
        if report.prediction.compliant:
            layers.add("prediction")
        if report.reasoning.compliant:
            layers.add("reasoning")
        if report.justification.compliant:
            layers.add("justification")
        if report.construction.compliant:
            layers.add("construction")
        return layers