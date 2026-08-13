# tests/phase14/test_validation_policy.py
"""
Phase 14.0C-2: ValidationPolicy 测试
"""

import pytest
from dataclasses import FrozenInstanceError
from src.writing.runtime import ValidationPolicy


class TestValidationPolicy:
    def test_development_policy(self):
        policy = ValidationPolicy.development()
        assert policy.allow_degraded_pass is True
        assert policy.max_retry == 3
        assert policy.fail_on_error is False
        assert policy.recovery_enabled is False

    def test_production_policy(self):
        policy = ValidationPolicy.production()
        assert policy.allow_degraded_pass is False
        assert policy.max_retry == 3
        assert policy.fail_on_error is True
        assert policy.recovery_enabled is False

    def test_custom_policy(self):
        policy = ValidationPolicy(
            allow_degraded_pass=True,
            max_retry=5,
            fail_on_error=False,
            recovery_enabled=True,
        )
        assert policy.allow_degraded_pass is True
        assert policy.max_retry == 5
        assert policy.fail_on_error is False
        assert policy.recovery_enabled is True

    def test_policy_is_immutable(self):
        policy = ValidationPolicy.production()
        with pytest.raises(FrozenInstanceError):
            policy.max_retry = 5  # type: ignore

    def test_profiles_are_independent(self):
        dev = ValidationPolicy.development()
        prod = ValidationPolicy.production()
        assert dev.allow_degraded_pass != prod.allow_degraded_pass