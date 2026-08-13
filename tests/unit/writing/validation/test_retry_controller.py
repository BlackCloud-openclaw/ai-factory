import pytest
from src.writing.validation.models import MissingContractChange, ContractSeverity
from src.writing.validation.retry_controller import ContractRetryController
from src.writing.runtime.validation_policy import ValidationPolicy
from src.writing.runtime.enforcement_mode import EnforcementMode
from src.writing.validation.feedback import ValidationFeedbackCompiler


class TestContractRetryController:
    def setup_method(self):
        self.controller = ContractRetryController(
            feedback_compiler=ValidationFeedbackCompiler(max_items=3)
        )
        self.policy_observe = ValidationPolicy(enforcement_mode=EnforcementMode.OBSERVE, max_retry=2)
        self.policy_retry = ValidationPolicy(enforcement_mode=EnforcementMode.RETRY, max_retry=2)
        self.policy_strict = ValidationPolicy(enforcement_mode=EnforcementMode.STRICT, max_retry=2)

    def test_no_missing(self):
        decision = self.controller.decide([], 0, self.policy_retry)
        assert not decision.should_retry
        assert decision.reason == "No valid missing changes"

    def test_only_warning(self):
        changes = [
            MissingContractChange(
                type="warning",
                description="Warning only",
                severity=ContractSeverity.WARNING,
            )
        ]
        decision = self.controller.decide(changes, 0, self.policy_retry)
        assert not decision.should_retry
        assert "WARNING-level" in decision.reason

    def test_blocking_observe_mode(self):
        changes = [
            MissingContractChange(
                type="plot_flag",
                description="Missing flag",
                severity=ContractSeverity.BLOCKING,
            )
        ]
        decision = self.controller.decide(changes, 0, self.policy_observe)
        assert not decision.should_retry
        assert "does not allow retry" in decision.reason   # 改为检查实际文本
        assert decision.writing_feedback == ""

    def test_blocking_retry_budget_available(self):
        changes = [
            MissingContractChange(
                type="plot_flag",
                description="Missing flag",
                severity=ContractSeverity.BLOCKING,
            )
        ]
        decision = self.controller.decide(changes, 0, self.policy_retry)
        assert decision.should_retry
        assert decision.next_retry_count == 1
        assert decision.writing_feedback != ""
        assert "BLOCKING missing changes" in decision.reason
        assert "retry 1/2" in decision.reason

    def test_blocking_retry_budget_exhausted(self):
        changes = [
            MissingContractChange(
                type="plot_flag",
                description="Missing flag",
                severity=ContractSeverity.BLOCKING,
            )
        ]
        decision = self.controller.decide(changes, 2, self.policy_retry)
        assert not decision.should_retry
        assert "reached max" in decision.reason

    def test_blocking_last_retry(self):
        changes = [
            MissingContractChange(
                type="plot_flag",
                description="Missing flag",
                severity=ContractSeverity.BLOCKING,
            )
        ]
        decision = self.controller.decide(changes, 1, self.policy_retry)
        assert decision.should_retry
        assert decision.next_retry_count == 2
        assert "retry 2/2" in decision.reason

    def test_blocking_strict_retry_available(self):
        changes = [
            MissingContractChange(
                type="plot_flag",
                description="Missing flag",
                severity=ContractSeverity.BLOCKING,
            )
        ]
        decision = self.controller.decide(changes, 0, self.policy_strict)
        assert decision.should_retry

    def test_blocking_strict_exhausted(self):
        changes = [
            MissingContractChange(
                type="plot_flag",
                description="Missing flag",
                severity=ContractSeverity.BLOCKING,
            )
        ]
        decision = self.controller.decide(changes, 2, self.policy_strict)
        assert not decision.should_retry
        assert "reached max" in decision.reason

    def test_dict_input_normalization(self):
        changes = [
            {
                "type": "plot_flag",
                "description": "Missing flag",
                "severity": "blocking",
                "actor": None,
                "fields": {},
                "source": "planning_contract",
                "contract_id": "scene_1",
                "confidence": 1.0,
            }
        ]
        decision = self.controller.decide(changes, 0, self.policy_retry)
        assert decision.should_retry
        assert decision.next_retry_count == 1

    def test_invalid_dict_skip(self):
        changes = [
            {"invalid": "data"},
            {
                "type": "plot_flag",
                "description": "Valid",
                "severity": "blocking",
            },
        ]
        decision = self.controller.decide(changes, 0, self.policy_retry)
        # 应跳过第一个非法项，使用第二个有效项
        assert decision.should_retry