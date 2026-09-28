"""
Phase 15.3 — Shadow Validator

Validator 实现：使用生产 SemanticValidator 验证 Shadow Rewrite 结果。
"""

from dataclasses import dataclass
from typing import Any, List

from .runner import Validator
from src.writing.planning_contract import PlanningContract
from src.writing.validation.semantic_validator import SemanticValidator


@dataclass(frozen=True)
class ShadowValidationResult:
    """
    Shadow Validator 对 Runner 暴露的最小结果协议。

    Runner 当前通过属性访问：
        result.passed
        result.violations

    因此不能返回 dict。
    """

    passed: bool
    violations: List[Any]
    raw_result: Any


class ShadowValidator(Validator):
    """
    使用生产 SemanticValidator 验证 Rewrite 结果。

    注意：
    - 不验证 Original（由 2B 从生产路径传入）
    - 只验证 Rewrite 文本
    - 不依赖 AgentState
    """

    def __init__(self, validator: SemanticValidator):
        self._validator = validator

    async def validate(self, text: str, contract: Any) -> ShadowValidationResult:
        """
        验证 Rewrite 文本。

        Args:
            text: 待验证的文本
            contract: PlanningContract（dict 或对象）

        Returns:
            ShadowValidationResult: 兼容 Runner 的验证结果
        """
        # 确保 contract 是 PlanningContract 对象
        if isinstance(contract, dict):
            contract = PlanningContract.model_validate(contract)

        result = self._validator.validate(contract, text)

        # 收集 violations：保留所有失败信号
        violations = list(result.missing or [])
        if result.errors:
            violations.extend(result.errors)

        return ShadowValidationResult(
            passed=result.passed,
            violations=violations,
            raw_result=result,
        )