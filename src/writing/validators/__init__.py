# src/writing/validators/__init__.py
"""验证器注册表 - 使用语义验证替代精确匹配"""
from typing import Dict, Any, Tuple, Optional, List

from .base import BaseValidator
from .structural import StructuralValidator
from .safety import SafetyValidator
from .rule_based import RuleBasedValidator
from .semantic import SemanticValidator

# 同步验证器列表（结构验证和安全验证必须同步执行）
SYNC_VALIDATORS: List[BaseValidator] = [
    StructuralValidator(),   # JSON 格式验证（致命）
    SafetyValidator(),       # 安全/注入验证（致命）
]

# 语义验证器（异步，用于 must_events 检查）
SEMANTIC_VALIDATOR = SemanticValidator()


def validate_all(
    text: str,
    context: Dict[str, Any],
    async_semantic: bool = False,
) -> Dict[str, Any]:
    """
    执行同步验证器，并标记是否需要异步语义验证
    """
    parsed = None
    
    # 1. 执行同步验证
    for validator in SYNC_VALIDATORS:
        passed, error = validator.validate(text, context)
        if not passed:
            return {
                "passed": False,
                "error": f"{validator.__class__.__name__}: {error}",
                "parsed_output": context.get("parsed_output"),
                "need_semantic": False,
            }
    
    # 2. 如果有必须事件，标记需要语义验证（不再使用 RuleBasedValidator 的精确匹配）
    must_events = context.get("must_events", [])
    if must_events:
        return {
            "passed": True,  # 同步验证通过，但需要异步语义验证
            "error": None,
            "parsed_output": context.get("parsed_output"),
            "need_semantic": True,
        }
    
    return {
        "passed": True,
        "error": None,
        "parsed_output": context.get("parsed_output"),
        "need_semantic": False,
    }


async def validate_semantic(
    text: str,
    context: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """
    异步语义验证 - 检查 must_events 是否在语义上被覆盖
    """
    must_events = context.get("must_events", [])
    if not must_events:
        return True, None
    
    parsed = context.get("parsed_output", {})
    scene_text = parsed.get("scene_text", text)
    
    # 使用语义验证器（embedding 相似度）
    return await SEMANTIC_VALIDATOR.validate_async(scene_text, context)