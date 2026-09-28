import pytest
from src.writing.shadow.runner import _normalize_validation_result


class FakeValidatorResult:
    def __init__(self, passed, violations=None, missing=None, errors=None):
        self.passed = passed
        self.violations = violations
        self.missing = missing
        self.errors = errors


class TestNormalizeValidationResult:
    def test_dict_with_violations(self):
        result = {"passed": True, "violations": ["error1", "error2"]}
        passed, violations = _normalize_validation_result(result)
        assert passed is True
        assert violations == ["error1", "error2"]

    def test_dict_with_missing(self):
        result = {"passed": False, "missing": ["missing_event"]}
        passed, violations = _normalize_validation_result(result)
        assert passed is False
        assert violations == ["missing_event"]

    def test_dict_with_errors(self):
        result = {"passed": False, "errors": ["syntax error"]}
        passed, violations = _normalize_validation_result(result)
        assert passed is False
        assert violations == ["syntax error"]

    def test_dict_empty(self):
        result = {}
        passed, violations = _normalize_validation_result(result)
        assert passed is False
        assert violations == []

    def test_object_with_violations(self):
        obj = FakeValidatorResult(passed=True, violations=["violation1"])
        passed, violations = _normalize_validation_result(obj)
        assert passed is True
        assert violations == ["violation1"]

    def test_object_with_missing(self):
        obj = FakeValidatorResult(passed=False, missing=["missing_event"])
        passed, violations = _normalize_validation_result(obj)
        assert passed is False
        assert violations == ["missing_event"]

    def test_object_with_errors(self):
        obj = FakeValidatorResult(passed=False, errors=["critical error"])
        passed, violations = _normalize_validation_result(obj)
        assert passed is False
        assert violations == ["critical error"]

    def test_object_empty(self):
        obj = FakeValidatorResult(passed=False)
        passed, violations = _normalize_validation_result(obj)
        assert passed is False
        assert violations == []