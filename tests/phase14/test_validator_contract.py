# tests/phase14/test_validator_contract.py
"""
Phase 14.0C-2: Validator Contract Freeze 测试

验证 ValidatorOutput 协议的稳定性。
"""

import pytest
from src.writing.validation_result import (
    ValidatorOutput,
    Violation,
    ViolationSeverity,
    ValidationStage,
    ValidationStatus,
)


class TestValidatorContract:
    def test_success_construction(self):
        """测试成功构造"""
        output = ValidatorOutput.success("exec_001", ValidationStage.CONTRACT)
        assert output.execution_id == "exec_001"
        assert output.stage == ValidationStage.CONTRACT
        assert output.status == ValidationStatus.PASSED
        assert output.valid is True
        assert output.is_degraded is False
        assert len(output.violations) == 0
        assert output.confidence == 1.0

    def test_degraded_construction(self):
        """测试降级通过构造"""
        warning = Violation(
            rule_id="LOW_CONFIDENCE",
            severity=ViolationSeverity.WARNING,
            description="Confidence below threshold"
        )
        output = ValidatorOutput.degraded(
            "exec_002",
            [warning],
            stage=ValidationStage.SEMANTIC,
            confidence=0.7
        )
        assert output.execution_id == "exec_002"
        assert output.stage == ValidationStage.SEMANTIC
        assert output.status == ValidationStatus.DEGRADED
        assert output.valid is True
        assert output.is_degraded is True
        assert len(output.violations) == 1
        assert output.violations[0].severity == ViolationSeverity.WARNING
        assert output.confidence == 0.7

    def test_failure_construction(self):
        """测试失败构造"""
        error = Violation(
            rule_id="STATE_CHANGE_MISSING",
            severity=ViolationSeverity.ERROR,
            description="Required state change missing",
            context="realm_change",
            location="Scene 2"
        )
        output = ValidatorOutput.failure(
            "exec_003",
            [error],
            stage=ValidationStage.CONTRACT,
            repaired_output='{"scene_text": "..."}'
        )
        assert output.execution_id == "exec_003"
        assert output.stage == ValidationStage.CONTRACT
        assert output.status == ValidationStatus.FAILED
        assert output.valid is False
        assert output.is_degraded is False
        assert len(output.violations) == 1
        assert output.violations[0].rule_id == "STATE_CHANGE_MISSING"
        assert output.violations[0].severity == ViolationSeverity.ERROR
        assert output.violations[0].context == "realm_change"
        assert output.repaired_output == '{"scene_text": "..."}'
        assert output.confidence == 0.0

    def test_to_runtime_dict(self):
        """测试 to_runtime_dict 包含兼容字段"""
        output = ValidatorOutput.success("exec_004")
        runtime_dict = output.to_runtime_dict()
        # 包含基础字段
        assert "execution_id" in runtime_dict
        assert "status" in runtime_dict
        assert "violations" in runtime_dict
        # 包含兼容字段
        assert runtime_dict["valid"] is True
        assert runtime_dict["passed"] is True

    def test_schema_status_enum(self):
        """验证 status 枚举值稳定"""
        # 直接检查枚举值，避免依赖 Pydantic 内部 JSON Schema 结构
        expected_statuses = {"passed", "degraded", "failed"}
        actual_statuses = {s.value for s in ValidationStatus}
        assert actual_statuses == expected_statuses

    def test_violation_serialization(self):
        """测试 Violation 序列化"""
        violation = Violation(
            rule_id="TEST_RULE",
            severity=ViolationSeverity.ERROR,
            description="Test violation"
        )
        data = violation.model_dump()
        assert data["rule_id"] == "TEST_RULE"
        assert data["severity"] == "error"
        assert data["description"] == "Test violation"
        assert data.get("context") is None
        assert data.get("location") is None