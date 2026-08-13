import pytest
from dataclasses import FrozenInstanceError
from uuid import UUID

from src.narrative import (
    ValidationSeverity,
    ValidationDomain,
    ValidationItem,
    ValidationResult,
)


class TestValidationItem:
    def test_immutable(self):
        item = ValidationItem(
            domain=ValidationDomain.RUNTIME,
            dimension="event_order",
            passed=True,
            severity=ValidationSeverity.PASS,
            message="All events in order",
        )
        with pytest.raises(FrozenInstanceError):
            item.message = "changed"  # type: ignore

    def test_serialization_roundtrip(self):
        original = ValidationItem(
            domain=ValidationDomain.NARRATIVE,
            dimension="dialogue_coverage",
            passed=False,
            severity=ValidationSeverity.WARNING,
            message="Dialogue coverage below threshold",
            evidence="Scene 2 has no dialogue",
        )
        data = original.to_dict()
        restored = ValidationItem.from_dict(data)

        assert original.domain == restored.domain
        assert original.dimension == restored.dimension
        assert original.passed == restored.passed
        assert original.severity == restored.severity
        assert original.message == restored.message
        assert original.evidence == restored.evidence

    def test_enum_serialization(self):
        item = ValidationItem(
            domain=ValidationDomain.RUNTIME,
            dimension="test",
            passed=True,
            severity=ValidationSeverity.PASS,
            message="ok",
        )
        data = item.to_dict()
        assert data["domain"] == "runtime"
        assert data["severity"] == "pass"

        restored = ValidationItem.from_dict(data)
        assert restored.domain == ValidationDomain.RUNTIME
        assert restored.severity == ValidationSeverity.PASS


class TestValidationResult:
    def test_immutable(self):
        result = ValidationResult(passed=True)
        with pytest.raises(FrozenInstanceError):
            result.passed = False  # type: ignore

    def test_serialization_roundtrip(self):
        item = ValidationItem(
            domain=ValidationDomain.RUNTIME,
            dimension="event_order",
            passed=True,
            severity=ValidationSeverity.PASS,
            message="All events in order",
        )
        original = ValidationResult(
            passed=True,
            items=(item,),
            summary={"total": 1, "passed": 1},
        )
        data = original.to_dict()
        restored = ValidationResult.from_dict(data)

        assert str(original.validation_id) == str(restored.validation_id)
        assert original.passed == restored.passed
        assert len(restored.items) == 1
        assert restored.items[0].message == "All events in order"

    def test_success_factory(self):
        result = ValidationResult.success("Everything is fine")
        assert result.passed is True
        assert len(result.items) == 1
        assert result.items[0].severity == ValidationSeverity.PASS
        assert result.items[0].message == "Everything is fine"
        assert result.items[0].domain == ValidationDomain.SYSTEM

    def test_failure_factory(self):
        result = ValidationResult.failure("Something went wrong")
        assert result.passed is False
        assert len(result.items) == 1
        assert result.items[0].severity == ValidationSeverity.ERROR
        assert result.items[0].message == "Something went wrong"
        assert result.items[0].domain == ValidationDomain.SYSTEM

    def test_uuid_generation(self):
        r1 = ValidationResult(passed=True)
        r2 = ValidationResult(passed=True)
        assert r1.validation_id != r2.validation_id
        assert isinstance(r1.validation_id, UUID)

    def test_bool_and_len(self):
        result1 = ValidationResult(passed=True)
        result2 = ValidationResult(passed=False)

        assert bool(result1) is True
        assert bool(result2) is False
        assert len(result1) == 0
        assert len(result2) == 0

        item = ValidationItem(
            domain=ValidationDomain.RUNTIME,
            dimension="test",
            passed=True,
            severity=ValidationSeverity.PASS,
            message="ok",
        )
        result3 = ValidationResult(passed=True, items=(item,))
        assert len(result3) == 1

    def test_schema_version(self):
        assert ValidationItem.SCHEMA_VERSION == "1.0.0"
        assert ValidationResult.SCHEMA_VERSION == "1.0.0"