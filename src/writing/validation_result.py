# src/writing/validation_result.py
"""
Phase 14.0C-2: Validator Output Contract

冻结 Validator 输出协议，作为 Runtime 控制平面的一部分。
"""

from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field


class ValidationStage(str, Enum):
    """验证阶段 - 用于识别不同类型的验证"""
    CONTRACT = "contract"          # 契约结构验证
    SEMANTIC = "semantic"          # 语义验证
    CONTINUITY = "continuity"      # 连续性验证
    STYLE = "style"                # 风格验证


class ViolationSeverity(str, Enum):
    """违规严重程度"""
    ERROR = "error"
    WARNING = "warning"


class ValidationStatus(str, Enum):
    """验证状态 - Runtime 控制的主要依据"""
    PASSED = "passed"              # 完全通过
    DEGRADED = "degraded"          # 有警告但通过
    FAILED = "failed"              # 阻断性错误


class Violation(BaseModel):
    """单个违规项"""
    rule_id: str = Field(..., description="规则标识，如 'STATE_CHANGE_MISSING'")
    severity: ViolationSeverity = Field(..., description="严重程度")
    description: str = Field(..., description="违规描述")
    context: Optional[str] = Field(None, description="上下文信息，如涉及的事件或字段")
    location: Optional[str] = Field(None, description="违规发生位置，如行号或JSON路径")


class ValidatorOutput(BaseModel):
    """
    Validator 输出协议 - 冻结的接口。

    所有 Validator Agent 必须输出此格式。
    这是 Validator 与 Runtime 之间的契约。
    """
    execution_id: str = Field(..., description="关联的 Execution ID，用于 Audit 追踪")
    stage: ValidationStage = Field(..., description="验证阶段")
    status: ValidationStatus = Field(..., description="验证状态")
    violations: List[Violation] = Field(default_factory=list, description="违规列表")
    repaired_output: Optional[str] = Field(None, description="修复后的输出（仅当 Validator 支持修复时）")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="验证置信度，默认 0.0 表示不确定")

    @property
    def valid(self) -> bool:
        """便捷属性：是否通过（PASSED 或 DEGRADED 均为通过）"""
        return self.status in (ValidationStatus.PASSED, ValidationStatus.DEGRADED)

    @property
    def is_degraded(self) -> bool:
        """是否降级通过"""
        return self.status == ValidationStatus.DEGRADED

    @classmethod
    def success(cls, execution_id: str, stage: ValidationStage = ValidationStage.SEMANTIC) -> "ValidatorOutput":
        """快捷构造通过结果（置信度 1.0）"""
        return cls(
            execution_id=execution_id,
            stage=stage,
            status=ValidationStatus.PASSED,
            violations=[],
            confidence=1.0
        )

    @classmethod
    def degraded(
        cls,
        execution_id: str,
        warnings: List[Violation],
        stage: ValidationStage = ValidationStage.SEMANTIC,
        confidence: float = 0.8
    ) -> "ValidatorOutput":
        """快捷构造降级通过结果"""
        return cls(
            execution_id=execution_id,
            stage=stage,
            status=ValidationStatus.DEGRADED,
            violations=warnings,
            confidence=confidence
        )

    @classmethod
    def failure(
        cls,
        execution_id: str,
        errors: List[Violation],
        stage: ValidationStage = ValidationStage.SEMANTIC,
        repaired_output: Optional[str] = None
    ) -> "ValidatorOutput":
        """快捷构造失败结果（置信度 0.0）"""
        return cls(
            execution_id=execution_id,
            stage=stage,
            status=ValidationStatus.FAILED,
            violations=errors,
            repaired_output=repaired_output,
            confidence=0.0
        )

    def to_runtime_dict(self) -> dict:
        """
        为 Runtime 提供兼容性字典，包含 'valid' 和 'passed' 字段。
        用于 Validator Agent 与 validate_node 之间的过渡期。
        """
        return {
            **self.model_dump(),
            "valid": self.valid,
            "passed": self.valid,
        }
      
        
class ValidationResult:
    """
    契约验证专用结果类（与 SemanticValidator 的 ValidationResult 区分）。
    用于 ContractConsistencyValidator 和 StateChangeValidator。
    """
    def __init__(
        self,
        valid: bool = True,
        checked_count: int = 0,
        errors: List[str] = None,
        warnings: List[str] = None,
    ):
        self.valid = valid
        self.checked_count = checked_count
        self.errors = errors or []
        self.warnings = warnings or []

    def add_error(self, error: str) -> None:
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)

    @classmethod
    def success(cls, checked_count: int = 0) -> "ValidationResult":
        return cls(valid=True, checked_count=checked_count)

    @classmethod
    def failure(
        cls,
        errors: List[str],
        warnings: List[str] = None,
        checked_count: int = 0
    ) -> "ValidationResult":
        result = cls(valid=False, checked_count=checked_count)
        result.errors = errors
        result.warnings = warnings or []
        return result