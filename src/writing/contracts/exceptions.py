# src/writing/contracts/exceptions.py
"""
Phase 14.0B-2: Contract 验证异常（增强诊断版）
"""

from typing import List, Dict, Any, Optional

# 为避免循环导入，在方法内延迟导入，或使用 TYPE_CHECKING
# 这里直接导入 ValidationResult，因为它在同一层级且无循环依赖
from src.writing.validation_result import ValidationResult


class InvalidSceneContract(Exception):
    """
    场景 Contract 无效异常。

    由 SceneEventValidator 抛出，触发 Planner retry。
    异常包含需要修复的具体事件列表。
    """

    def __init__(
        self,
        scene_index: int,
        invalid_events: List[str],
        validation_result: Any = None,
        message: str = None,
    ):
        self.scene_index = scene_index
        self.invalid_events = invalid_events
        self.validation_result = validation_result

        # 自动生成详细消息（如果未提供）
        if message is None:
            lines = [f"Scene {scene_index} has {len(invalid_events)} invalid event(s):"]
            for evt in invalid_events[:5]:
                lines.append(f"  - {evt}")
            if len(invalid_events) > 5:
                lines.append(f"  ... and {len(invalid_events)-5} more")
            if validation_result and hasattr(validation_result, 'summary'):
                lines.append(f"  Summary: {validation_result.summary}")
            message = "\n".join(lines)

        self.message = message
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_index": self.scene_index,
            "invalid_events": self.invalid_events,
            "message": self.message,
        }


class ContractValidationError(Exception):
    """
    Contract 一致性验证失败异常。

    由 ContractConsistencyValidator 抛出，触发 Planner retry。
    包含所有验证错误和警告。
    """

    def __init__(
        self,
        scene_id: str,
        errors: List[str],
        warnings: List[str] = None,
        message: str = None,
    ):
        self.scene_id = scene_id
        self.errors = errors
        self.warnings = warnings or []

        # 自动生成详细消息（如果未提供）
        if message is None:
            lines = [f"Contract '{scene_id}' validation failed: {len(errors)} error(s)"]
            if self.warnings:
                lines.append(f"  Warnings: {len(self.warnings)}")
            if errors:
                lines.append("  Errors:")
                for e in errors:
                    lines.append(f"    - {e}")
            if self.warnings:
                lines.append("  Warnings:")
                # 显示前3个警告，避免过长
                for w in self.warnings[:3]:
                    lines.append(f"    - {w}")
                if len(self.warnings) > 3:
                    lines.append(f"    ... and {len(self.warnings)-3} more")
            message = "\n".join(lines)

        self.message = message
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "errors": self.errors,
            "warnings": self.warnings,
            "message": self.message,
        }

    @classmethod
    def from_validation_result(cls, scene_id: str, result: ValidationResult) -> "ContractValidationError":
        """从 ValidationResult 创建异常，自动生成诊断消息"""
        return cls(
            scene_id=scene_id,
            errors=result.errors,
            warnings=result.warnings,
            # 不传递 message，由 __init__ 自动生成
        )