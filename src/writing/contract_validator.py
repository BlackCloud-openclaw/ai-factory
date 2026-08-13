# src/writing/contract_validator.py
"""
Phase 14.0B: Contract Validation & StateChange Consistency Gate
"""

from typing import List
import logging

from src.writing.planning_contract import (
    PlanningContract,
    StateChange,
    SignalSource,
)
from src.writing.state_change_types import StateChangeType
from src.writing.validation_result import ValidationResult

logger = logging.getLogger(__name__)


class StateChangeValidator:
    """
    验证单个 StateChange 的结构完整性

    规则:
    - type 必须属于 StateChangeType 枚举
    - source 必须为 LLM 或 INFERRED（不允许 UNKNOWN）
    - confidence 必须在 0.0 ~ 1.0 之间
    - confidence < 0.5 时产生警告
    """

    @classmethod
    def validate(cls, state_change: StateChange) -> ValidationResult:
        errors = []
        warnings = []

        # 1. type 合法性
        valid_types = StateChangeType.values()
        if state_change.type not in valid_types:
            errors.append(
                f"Invalid StateChange.type: '{state_change.type}', "
                f"must be one of {sorted(valid_types)}"
            )

        # 2. source 合法性
        if state_change.source not in (SignalSource.LLM, SignalSource.INFERRED):
            if state_change.source == SignalSource.UNKNOWN:
                errors.append(
                    f"StateChange.source is UNKNOWN, expected LLM or INFERRED "
                    f"(id={state_change.id})"
                )
            else:
                errors.append(
                    f"Invalid StateChange.source: '{state_change.source}', "
                    f"expected LLM or INFERRED (id={state_change.id})"
                )

        # 3. confidence 范围检查 + 低置信度警告
        if not hasattr(state_change, 'confidence') or state_change.confidence is None:
            errors.append(
                f"StateChange missing confidence field (id={state_change.id})"
            )
        else:
            confidence = state_change.confidence
            if not (0.0 <= confidence <= 1.0):
                errors.append(
                    f"StateChange.confidence out of range [0.0,1.0]: {confidence} "
                    f"(id={state_change.id})"
                )
            elif confidence < 0.5:
                warnings.append(
                    f"Low confidence: {confidence:.2f} (id={state_change.id})"
                )

        # 4. 检查 id 是否存在
        if not hasattr(state_change, 'id') or not state_change.id:
            errors.append("StateChange missing id field")

        # 返回结果：有错误时失败，否则成功但保留警告
        if errors:
            return ValidationResult.failure(errors, warnings, checked_count=1)
        return ValidationResult(valid=True, errors=[], warnings=warnings, checked_count=1)


class ContractConsistencyValidator:
    """
    验证 PlanningContract 的整体一致性

    规则:
    - 至少有一个 StateChange
    - 所有 StateChange ID 唯一
    - 每个 StateChange 都必须通过 StateChangeValidator
    - 不负责检测 LLM 信号的覆盖（由 Normalizer 保证）
    """
    
    @classmethod
    def validate(cls, contract: PlanningContract) -> ValidationResult:
        result = ValidationResult(valid=True, checked_count=0)
        state_changes = contract.observables.state_changes

        # ========== 强制检查：必须至少有一个 StateChange ==========
        if not state_changes:
            # 直接设置错误，不依赖 add_error（防御性）
            result.errors.append("StateChanges cannot be empty (minimum 1 required)")
            result.valid = False
            # 空列表无需继续检查，直接返回
            return result

        # Rule C2: ID 唯一性
        ids = [sc.id for sc in state_changes if hasattr(sc, 'id')]
        if len(ids) != len(set(ids)):
            duplicate_ids = [id for id in ids if ids.count(id) > 1]
            result.add_error(f"Duplicate StateChange ids: {duplicate_ids}")

        # Rule C3: 逐项验证
        for sc in state_changes:
            sc_result = StateChangeValidator.validate(sc)
            result.errors.extend(sc_result.errors)
            result.warnings.extend(sc_result.warnings)
            if not sc_result.valid:
                result.valid = False
            result.checked_count += 1

        # 如果 errors 非空，确保 valid=False
        if result.errors:
            result.valid = False

        return result

    @classmethod
    def validate_contract(cls, contract: PlanningContract) -> bool:
        """便捷方法，返回布尔值"""
        result = cls.validate(contract)
        return result.valid