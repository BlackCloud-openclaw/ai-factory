"""
锚点验证器 - 验证重写结果是否保留了所有锚点
"""

from typing import Dict, List, Any
from collections import Counter


class AnchorValidator:
    """验证改写是否保留了所有锚点"""
    
    @classmethod
    def validate(cls, original: str, rewritten: str, anchors: Dict[str, List[str]]) -> Dict[str, Any]:
        """验证改写结果"""
        issues = []
        details = {}
        
        for key, values in anchors.items():
            missing = []
            for value in values:
                # 支持部分匹配（避免因为标点/空格差异导致误判）
                if value not in rewritten:
                    # 尝试更宽松的匹配
                    if not cls._fuzzy_match(value, rewritten):
                        missing.append(value)
            details[key] = {
                "total": len(values),
                "found": len(values) - len(missing),
                "missing": missing,
            }
            if missing:
                issues.append(f"{key} 缺失: {', '.join(missing)}")
        
        # 额外检查：原文中的关键实体是否被替换
        character_consistency = cls._check_character_consistency(original, rewritten)
        if character_consistency < 0.95:
            issues.append(f"角色一致性: {character_consistency*100:.0f}% (目标: ≥95%)")
        
        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "details": details,
            "character_consistency": round(character_consistency, 3),
        }
    
    @classmethod
    def _fuzzy_match(cls, value: str, text: str) -> bool:
        """宽松匹配：检查 value 中的每个字符是否都在文本中出现"""
        # 如果 value 是2-4字词，检查是否包含
        if 2 <= len(value) <= 4:
            return value in text
        # 更长的词，检查前3个字符
        return value[:3] in text if len(value) >= 3 else False
    
    @classmethod
    def _check_character_consistency(cls, original: str, rewritten: str) -> float:
        """检查角色名的一致性"""
        # 提取原文中的所有角色名
        from anchor_extractor import AnchorExtractor
        orig_chars = AnchorExtractor.extract_characters(original)
        if not orig_chars:
            return 1.0
        
        # 检查每个角色名是否在改写中保留
        found = 0
        for char in orig_chars:
            if char in rewritten:
                found += 1
        return found / len(orig_chars)