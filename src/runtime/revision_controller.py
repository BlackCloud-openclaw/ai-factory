# src/runtime/revision_controller.py
"""
Revision Controller - 修订策略决策器 (确定性纯函数)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional

from src.runtime.failure_analyzer import FailureDiagnosis, FailureAnalysis, FailureType, Severity
from src.runtime.validator import ComplianceReport
from src.runtime.metadata import RuntimeMetadata


class RevisionStrategy(Enum):
    SKIP = "skip"
    INSERT = "insert"
    REWRITE = "rewrite"
    REPLACE = "replace"
    FULL_RETRY = "full_retry"


class RevisionRisk(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class RevisionDecision:
    """修订决策 IR - 可重放、无执行细节"""
    id: str
    should_revise: bool
    strategy: RevisionStrategy
    target_layers: List[str]
    preserve_layers: List[str]
    rationale: str
    confidence: float
    estimated_risk: RevisionRisk


class RevisionController:
    """
    修订控制器 v1.0 - 确定性纯函数
    """

    VERSION = "1.0"

    COMPLIANCE_SKIP_THRESHOLD = 0.80
    COMPLIANCE_REVISE_THRESHOLD = 0.65

    _counter = 0

    @classmethod
    def _next_id(cls) -> str:
        cls._counter += 1
        return f"RD{cls._counter:04d}"

    def decide(
        self,
        diagnosis: FailureDiagnosis,
        compliance: ComplianceReport,
        metadata: Optional[RuntimeMetadata] = None,
    ) -> RevisionDecision:
        """确定性的修订决策"""
        compliance_rate = compliance.compliance_rate

        # 1. 合规率足够高 → 跳过
        if compliance_rate >= self.COMPLIANCE_SKIP_THRESHOLD:
            return self._decision(
                should_revise=False,
                strategy=RevisionStrategy.SKIP,
                target_layers=[],
                preserve_layers=["prediction", "reasoning", "justification", "construction"],
                rationale=f"合规率 {compliance_rate:.2f} ≥ {self.COMPLIANCE_SKIP_THRESHOLD:.2f}，无需修订。",
                confidence=0.95,
                risk=RevisionRisk.LOW,
            )

        # 2. 无失败诊断 → 跳过
        if not diagnosis.analyses:
            return self._decision(
                should_revise=False,
                strategy=RevisionStrategy.SKIP,
                target_layers=[],
                preserve_layers=["prediction", "reasoning", "justification", "construction"],
                rationale="未检测到失败层。",
                confidence=0.90,
                risk=RevisionRisk.LOW,
            )

        # 3. 合规率极低 → 激进策略
        if compliance_rate < self.COMPLIANCE_REVISE_THRESHOLD:
            return self._handle_low_compliance(diagnosis, compliance, metadata)

        # 4. 中间区间 → 根据失败类型决定
        return self._handle_medium_compliance(diagnosis, compliance, metadata)

    def _handle_low_compliance(
        self,
        diagnosis: FailureDiagnosis,
        compliance: ComplianceReport,
        metadata: Optional[RuntimeMetadata],
    ) -> RevisionDecision:
        """处理合规率极低的情况 (< 0.65)"""
        critical_layers = [
            a for a in diagnosis.analyses
            if a.severity in (Severity.CRITICAL, Severity.HIGH)
        ]

        if critical_layers:
            return self._decision(
                should_revise=True,
                strategy=RevisionStrategy.FULL_RETRY,
                target_layers=[a.layer for a in critical_layers],
                preserve_layers=[],
                rationale=f"合规率 {compliance.compliance_rate:.2f} 低于阈值且存在严重失败，需要完全重试。",
                confidence=0.95,
                risk=RevisionRisk.CRITICAL,
            )

        # 一般失败 → 重写，但保护已合规的层
        preserve = self._get_compliant_layers(compliance)
        return self._decision(
            should_revise=True,
            strategy=RevisionStrategy.REWRITE,
            target_layers=[a.layer for a in diagnosis.analyses],
            preserve_layers=preserve,
            rationale=f"合规率 {compliance.compliance_rate:.2f} 低于阈值，执行重写。",
            confidence=0.85,
            risk=RevisionRisk.HIGH,
        )

    def _handle_medium_compliance(
        self,
        diagnosis: FailureDiagnosis,
        compliance: ComplianceReport,
        metadata: Optional[RuntimeMetadata],
    ) -> RevisionDecision:
        """处理中等合规率的情况 (0.65-0.80)"""
        # 分类失败类型
        has_contradiction = any(
            a.failure_type == FailureType.STATE_CONTRADICTS_POLICY
            for a in diagnosis.analyses
        )
        has_no_state = any(
            a.failure_type == FailureType.NO_STATE
            for a in diagnosis.analyses
        )
        has_wrong_layer = any(
            a.failure_type == FailureType.STATE_WRONG_LAYER
            for a in diagnosis.analyses
        )
        has_mention_only = any(
            a.failure_type == FailureType.STATE_MENTIONED_ONLY
            for a in diagnosis.analyses
        )

        preserve = self._get_compliant_layers(compliance)

        if has_contradiction:
            return self._decision(
                should_revise=True,
                strategy=RevisionStrategy.FULL_RETRY,
                target_layers=[a.layer for a in diagnosis.analyses if a.failure_type == FailureType.STATE_CONTRADICTS_POLICY],
                preserve_layers=[],
                rationale="State 与 Policy 冲突，必须完全重试。",
                confidence=0.90,
                risk=RevisionRisk.CRITICAL,
            )

        if has_no_state:
            return self._decision(
                should_revise=True,
                strategy=RevisionStrategy.INSERT,
                target_layers=[a.layer for a in diagnosis.analyses if a.failure_type == FailureType.NO_STATE],
                preserve_layers=preserve,
                rationale="State 完全缺失，插入是最安全的修复方式。",
                confidence=0.85,
                risk=RevisionRisk.LOW,
            )

        if has_wrong_layer or has_mention_only:
            return self._decision(
                should_revise=True,
                strategy=RevisionStrategy.REWRITE,
                target_layers=[a.layer for a in diagnosis.analyses if a.failure_type in (FailureType.STATE_WRONG_LAYER, FailureType.STATE_MENTIONED_ONLY)],
                preserve_layers=preserve,
                rationale="State 存在但使用不当，重写相关层。",
                confidence=0.75,
                risk=RevisionRisk.MEDIUM,
            )

        # 默认保守策略
        return self._decision(
            should_revise=True,
            strategy=RevisionStrategy.INSERT,
            target_layers=[a.layer for a in diagnosis.analyses],
            preserve_layers=preserve,
            rationale=f"合规率 {compliance.compliance_rate:.2f}，采用保守的插入策略。",
            confidence=0.60,
            risk=RevisionRisk.LOW,
        )

    def _get_compliant_layers(self, compliance: ComplianceReport) -> List[str]:
        """从 ComplianceReport 获取已合规的层"""
        layers = []
        if compliance.prediction.compliant:
            layers.append("prediction")
        if compliance.reasoning.compliant:
            layers.append("reasoning")
        if compliance.justification.compliant:
            layers.append("justification")
        if compliance.construction.compliant:
            layers.append("construction")
        return layers

    def _decision(
        self,
        should_revise: bool,
        strategy: RevisionStrategy,
        target_layers: List[str],
        preserve_layers: List[str],
        rationale: str,
        confidence: float,
        risk: RevisionRisk,
    ) -> RevisionDecision:
        return RevisionDecision(
            id=self._next_id(),
            should_revise=should_revise,
            strategy=strategy,
            target_layers=target_layers,
            preserve_layers=preserve_layers,
            rationale=rationale,
            confidence=confidence,
            estimated_risk=risk,
        )