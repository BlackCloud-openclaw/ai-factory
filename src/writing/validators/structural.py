"""结构性验证器 - 检查 JSON 格式和基本结构"""
import json
from typing import Dict, Any, Tuple, Optional
from .base import BaseValidator


class StructuralValidator(BaseValidator):
    """JSON 结构验证器"""
    
    fatal = True
    
    def validate(self, text: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """检查是否为有效的 JSON"""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            # 尝试提取 JSON 部分
            import re
            match = re.search(r'\{.*\}|\[.*\]', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except:
                    return False, f"Invalid JSON: {e}"
            else:
                return False, f"Invalid JSON: {e}"
        
        # 检查必要字段
        if not isinstance(data, dict):
            return False, "Output must be a JSON object"
        
        required_fields = ["scene_text"]
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        if not isinstance(data["scene_text"], str):
            return False, "scene_text must be a string"
        
        if len(data["scene_text"].strip()) < 10:
            return False, f"scene_text too short: {len(data['scene_text'])} chars"
        
        # 将解析后的数据存回 context 供后续验证器使用
        context["parsed_output"] = data
        return True, None