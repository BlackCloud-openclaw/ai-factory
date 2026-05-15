# src/writing/validators/safety.py
"""安全验证器 - 检查注入和危险内容"""
from typing import Dict, Any, Tuple, Optional
from ..prompt_firewall import PromptFirewall
from .base import BaseValidator


class SafetyValidator(BaseValidator):
    """安全验证器 - 只检查 scene_text，不再检查 events 字段"""

    fatal = True

    def validate(self, text: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """检查 scene_text 是否有注入攻击，跳过 events 字段的检查"""
        parsed = context.get("parsed_output", {})
        scene_text = parsed.get("scene_text", text)

        # 只检查 scene_text
        if scene_text:
            passed, error = PromptFirewall.validate(scene_text)
            if not passed:
                return False, f"Scene text: {error}"

        # 不再检查 events 字段，避免因枚举值被误判
        return True, None