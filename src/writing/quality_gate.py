# src/writing/quality_gate.py
"""
Quality Gate - 将 ValidationResult 转化为 Writer 控制信号

Phase 13.2.3C v1.1
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal, Optional

# ✅ 直接从子模块导入，避免循环导入
from .validation.evidence import ValidationResult


@dataclass
class QualityGateResult:
    """QualityGate 输出结果。"""
    decision: Literal["pass", "retry", "force_pass"]
    score: float
    feedback: str
    details: Dict[str, Any] = field(default_factory=dict)


class QualityGate:
    """质量门控 - 将验证结果转化为 Writer 可执行的控制信号。"""

    def __init__(
        self,
        pass_threshold: float = 0.8,
        retry_threshold: float = 0.5,
        max_retries: int = 2,
    ):
        self.pass_threshold = pass_threshold
        self.retry_threshold = retry_threshold
        self.max_retries = max_retries

    def evaluate(
        self,
        result: ValidationResult,
        retry_count: int,
        max_retries: Optional[int] = None,
    ) -> QualityGateResult:
        if max_retries is None:
            max_retries = self.max_retries

        score = self._compute_score(result)
        has_blocking = len(result.blocking_missing) > 0

        if score >= self.pass_threshold and not has_blocking:
            decision = "pass"
            feedback = "验证通过，质量良好。"
        elif has_blocking and retry_count < max_retries:
            decision = "retry"
            feedback = self._generate_retry_feedback(result, result.blocking_missing)
        elif score < self.retry_threshold and retry_count < max_retries:
            decision = "retry"
            feedback = "质量分数偏低，建议重新生成以提升连贯性和完成度。"
        elif retry_count >= max_retries:
            decision = "force_pass"
            feedback = "重试次数用尽，强制通过。请检查后续章节是否连贯。"
        else:
            decision = "force_pass"
            feedback = "强制通过（低质量但无法继续重试）。"

        return QualityGateResult(
            decision=decision,
            score=score,
            feedback=feedback,
            details={
                "matched_count": result.match_count,
                "missing_count": result.missing_count,
                "blocking_missing": result.blocking_missing[:5],
                "retry_count": retry_count,
                "max_retries": max_retries,
                "thresholds": {
                    "pass": self.pass_threshold,
                    "retry": self.retry_threshold,
                },
            },
        )

    def _compute_score(self, result: ValidationResult) -> float:
        if result.weight_applied > 0:
            return min(1.0, result.weight_applied)
        if result.overall_confidence > 0:
            return min(1.0, result.overall_confidence)
        total = result.match_count + result.missing_count
        if total == 0:
            return 1.0
        return result.match_count / total

    def _generate_retry_feedback(self, result: ValidationResult, blocking_missing: List[str]) -> str:
        if not blocking_missing:
            return "质量分数偏低，请重新生成场景，确保完成所有必须事件。"
        top_missing = blocking_missing[:3]
        missing_text = "、".join(top_missing)
        if len(blocking_missing) > 3:
            missing_text += f" 等 {len(blocking_missing)} 个事件"
        return f"缺失关键事件：{missing_text}。请在下一次生成中明确包含这些事件。"