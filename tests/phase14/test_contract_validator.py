# tests/phase14/test_contract_validator.py
"""
Phase 14.0B: Contract Validator Tests
"""

import pytest
from src.writing.contract_validator import (
    StateChangeValidator,
    ContractConsistencyValidator,
)
from src.writing.planning_contract import (
    StateChange,
    SignalSource,
    PlanningContract,
    Intent,
    Execution,
    Observables,
    ContractMetadata,
)
from src.writing.state_change_types import StateChangeType
from src.writing.validation_result import ValidationResult


# ============================================================================
# Test Helpers
# ============================================================================

def create_valid_state_change(
    sc_type: str = StateChangeType.KNOWLEDGE_GAIN.value,
    source: SignalSource = SignalSource.INFERRED,
    confidence: float = 0.95,
    sc_id: str = "sc_001",
) -> StateChange:
    """创建有效的 StateChange，但 confidence 必须在 [0,1] 内"""
    return StateChange(
        id=sc_id,
        type=sc_type,
        source=source,
        confidence=confidence,
        name="test_name",
        value=True,
    )


def create_test_contract(state_changes: list) -> PlanningContract:
    return PlanningContract(
        version="1.0",
        scene_id="test_scene",
        intent=Intent(goal="test", conflict="test", expected_outcome="test"),
        execution=Execution(units=[]),
        observables=Observables(state_changes=state_changes),
        constraints=[],
        metadata=ContractMetadata(chapter=1, scene_index=0),
    )


# ============================================================================
# Test StateChangeValidator
# ============================================================================

class TestStateChangeValidator:
    def test_valid_inferred(self):
        sc = create_valid_state_change()
        result = StateChangeValidator.validate(sc)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_valid_llm(self):
        sc = create_valid_state_change(source=SignalSource.LLM, confidence=1.0)
        result = StateChangeValidator.validate(sc)
        assert result.valid is True

    def test_valid_boundary_confidence(self):
        # 边界值 0.0 和 1.0 应通过，不报错
        sc = create_valid_state_change(confidence=0.0)
        result = StateChangeValidator.validate(sc)
        assert result.valid is True
        sc = create_valid_state_change(confidence=1.0)
        result = StateChangeValidator.validate(sc)
        assert result.valid is True

    def test_invalid_type(self):
        sc = create_valid_state_change(sc_type="invalid_type")
        result = StateChangeValidator.validate(sc)
        assert result.valid is False
        assert "Invalid StateChange.type" in result.errors[0]

    def test_invalid_source_unknown(self):
        sc = create_valid_state_change(source=SignalSource.UNKNOWN)
        result = StateChangeValidator.validate(sc)
        assert result.valid is False
        assert "UNKNOWN" in result.errors[0]

    def test_missing_confidence(self):
        sc = create_valid_state_change()
        del sc.confidence
        result = StateChangeValidator.validate(sc)
        assert result.valid is False
        assert "missing confidence" in result.errors[0]

    def test_missing_id(self):
        sc = create_valid_state_change()
        del sc.id
        result = StateChangeValidator.validate(sc)
        assert result.valid is False
        assert "missing id" in result.errors[0]

    def test_low_confidence_warning(self):
        sc = create_valid_state_change(confidence=0.4)
        result = StateChangeValidator.validate(sc)
        assert result.valid is True
        # 应该有一个警告
        assert len(result.warnings) == 1
        assert "Low confidence" in result.warnings[0]

    def test_valid_enum_type(self):
        for state_type in StateChangeType:
            sc = create_valid_state_change(sc_type=state_type.value)
            result = StateChangeValidator.validate(sc)
            assert result.valid is True

    def test_validator_does_not_modify_input(self):
        sc = create_valid_state_change()
        before = sc.model_dump()
        StateChangeValidator.validate(sc)
        after = sc.model_dump()
        assert before == after


# ============================================================================
# Test ContractConsistencyValidator
# ============================================================================

class TestContractConsistencyValidator:
    def test_valid_contract(self):
        sc = create_valid_state_change()
        contract = create_test_contract([sc])
        result = ContractConsistencyValidator.validate(contract)
        assert result.valid is True
        assert result.checked_count == 1

    def test_empty_state_changes_fails(self):
        contract = create_test_contract([])
        result = ContractConsistencyValidator.validate(contract)
        assert result.valid is False
        assert "no state_changes" in result.errors[0].lower()

    def test_duplicate_ids_fails(self):
        sc1 = create_valid_state_change(sc_id="same_id")
        sc2 = create_valid_state_change(sc_id="same_id")
        contract = create_test_contract([sc1, sc2])
        result = ContractConsistencyValidator.validate(contract)
        assert result.valid is False
        assert "duplicate" in result.errors[0].lower()

    def test_mixed_llm_inferred_passes(self):
        # 分配不同的 ID
        sc1 = create_valid_state_change(sc_id="sc_llm", source=SignalSource.LLM, confidence=1.0)
        sc2 = create_valid_state_change(sc_id="sc_inf", source=SignalSource.INFERRED, confidence=0.9)
        contract = create_test_contract([sc1, sc2])
        result = ContractConsistencyValidator.validate(contract)
        assert result.valid is True
        assert result.checked_count == 2

    def test_all_inferred_does_not_warn(self):
        sc = create_valid_state_change(source=SignalSource.INFERRED)
        contract = create_test_contract([sc])
        result = ContractConsistencyValidator.validate(contract)
        assert result.valid is True
        # 不再有 "all inferred" 警告
        for w in result.warnings:
            assert "all state_changes are inferred" not in w.lower()

    def test_contract_validation_shortcut(self):
        sc = create_valid_state_change()
        contract = create_test_contract([sc])
        assert ContractConsistencyValidator.validate_contract(contract) is True

    def test_invalid_state_change_in_contract(self):
        sc = create_valid_state_change(sc_type="invalid_type")
        contract = create_test_contract([sc])
        result = ContractConsistencyValidator.validate(contract)
        assert result.valid is False
        assert "Invalid StateChange.type" in result.errors[0]

    def test_multiple_invalid_state_changes(self):
        sc1 = create_valid_state_change(sc_type="invalid_type")
        sc2 = create_valid_state_change(source=SignalSource.UNKNOWN)
        contract = create_test_contract([sc1, sc2])
        result = ContractConsistencyValidator.validate(contract)
        assert result.valid is False
        assert len(result.errors) >= 2

    def test_warnings_aggregated(self):
        sc = create_valid_state_change(confidence=0.3)
        contract = create_test_contract([sc])
        result = ContractConsistencyValidator.validate(contract)
        assert result.valid is True
        # 应该有一个警告
        assert len(result.warnings) == 1
        assert "Low confidence" in result.warnings[0]


# ============================================================================
# Test StateChangeType Enum
# ============================================================================

class TestStateChangeTypeEnum:
    def test_enum_count(self):
        assert len(StateChangeType) == 6

    def test_enum_values_stable(self):
        expected = {
            "knowledge_gain",
            "inventory_acquire",
            "location_change",
            "realm_change",
            "relationship_change",
            "plot_flag",
        }
        assert set(StateChangeType.values()) == expected

    def test_no_extra_values(self):
        assert len(StateChangeType) == 6


# ============================================================================
# Test ValidationResult
# ============================================================================

class TestValidationResult:
    def test_success_factory(self):
        result = ValidationResult.success(checked_count=5)
        assert result.valid is True
        assert result.checked_count == 5

    def test_failure_factory(self):
        result = ValidationResult.failure(
            errors=["e1", "e2"],
            warnings=["w1"],
            checked_count=3,
        )
        assert result.valid is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1

    def test_add_error(self):
        result = ValidationResult(valid=True)
        result.add_error("something wrong")
        assert result.valid is False
        assert "something wrong" in result.errors

    def test_add_warning(self):
        result = ValidationResult(valid=True)
        result.add_warning("something to note")
        assert result.valid is True
        assert "something to note" in result.warnings