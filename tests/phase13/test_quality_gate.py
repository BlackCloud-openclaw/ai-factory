# tests/phase13/test_quality_gate.py
import pytest
from src.writing.quality_gate import QualityGate
from src.writing.validation import ValidationResult, ValidationEvidence, SignalSource
# 或直接导入 SignalSource（已从 validation 导出）

# 其余内容不变（如之前提供）

def create_evidence(confidence=1.0, source=SignalSource.LLM, weight=1.0):
    return ValidationEvidence(
        evidence_id="e1",
        event_id="evt1",
        event_text="获得玉佩",
        matcher="exact",
        confidence=confidence,
        source=source,
        matched_text="获得玉佩",
        weight=weight,
    )


class TestQualityGate:
    def test_pass_decision(self):
        evidence = create_evidence()
        result = ValidationResult(
            passed=True,
            missing=[],
            matched=[evidence],
            blocking_missing=[],
            overall_confidence=1.0,
            weight_applied=1.0,
        )
        gate = QualityGate()
        res = gate.evaluate(result, retry_count=0)
        assert res.decision == "pass"
        assert res.score >= 0.8
        assert "通过" in res.feedback

    def test_retry_on_blocking_missing(self):
        result = ValidationResult(
            passed=False,
            missing=["获得玉佩"],
            matched=[],
            blocking_missing=["获得玉佩"],
            overall_confidence=0.0,
            weight_applied=0.0,
            errors=["Blocking missing: 获得玉佩"],
        )
        gate = QualityGate()
        res = gate.evaluate(result, retry_count=0, max_retries=2)
        assert res.decision == "retry"
        assert "缺失关键事件" in res.feedback

    def test_force_pass_on_retry_exhausted(self):
        result = ValidationResult(
            passed=False,
            missing=["获得玉佩"],
            matched=[],
            blocking_missing=["获得玉佩"],
            overall_confidence=0.0,
            weight_applied=0.0,
            errors=["Blocking missing: 获得玉佩"],
        )
        gate = QualityGate()
        res = gate.evaluate(result, retry_count=2, max_retries=2)
        assert res.decision == "force_pass"
        assert "重试次数用尽" in res.feedback

    def test_retry_on_low_score(self):
        evidence = create_evidence(confidence=0.3, source=SignalSource.INFERRED, weight=0.18)
        result = ValidationResult(
            passed=False,
            missing=["缺失事件"],
            matched=[evidence],
            blocking_missing=[],
            overall_confidence=0.3,
            weight_applied=0.18,
        )
        gate = QualityGate(pass_threshold=0.8, retry_threshold=0.5)
        res = gate.evaluate(result, retry_count=0, max_retries=2)
        assert res.decision == "retry"
        assert "质量分数偏低" in res.feedback

    def test_score_computation(self):
        # 没有 matched
        result = ValidationResult(
            passed=False,
            missing=["事件1", "事件2"],
            matched=[],
            blocking_missing=[],
            overall_confidence=0.0,
            weight_applied=0.0,
        )
        gate = QualityGate()
        score = gate._compute_score(result)
        assert score == 0.0

        # 部分匹配
        evidence = create_evidence()
        result = ValidationResult(
            passed=False,
            missing=["事件2"],
            matched=[evidence],
            blocking_missing=[],
            overall_confidence=0.5,
            weight_applied=0.5,
        )
        score = gate._compute_score(result)
        assert score == 0.5