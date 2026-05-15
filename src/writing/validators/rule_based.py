"""规则验证器 - 检查必须事件是否在正文中出现"""
import re
from typing import Dict, Any, Tuple, Optional, List
from .base import BaseValidator


class RuleBasedValidator(BaseValidator):
    """基于规则的验证器"""
    
    fatal = False
    
    def validate(self, text: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """检查 must_events 是否在正文中出现"""
        must_events = context.get("must_events", [])
        if not must_events:
            return True, None
        
        parsed = context.get("parsed_output", {})
        scene_text = parsed.get("scene_text", text)
        
        missing = []
        for event in must_events:
            # 提取关键词（简单策略：取前 10 个字符）
            if len(event) <= 10:
                keyword = event
            else:
                keyword = event[:10]
            
            if keyword not in scene_text:
                missing.append(event)
        
        if missing:
            return False, f"Missing must_events: {', '.join(missing)}"
        
        return True, None