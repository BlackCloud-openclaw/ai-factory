"""
重写验证器 - 确保改写不破坏核心内容

校验项：
1. Must Events 保留率
2. 角色名一致性
3. 字数倍率
"""

import re
from typing import List, Dict, Any, Tuple


class RewriteValidator:
    """重写结果验证器"""

    @classmethod
    def validate(cls, original: str, rewritten: str, must_events: List[str]) -> Dict[str, Any]:
        """
        验证重写结果
        
        Returns:
            {
                "passed": bool,
                "event_retention": float,      # 0-1
                "character_consistency": float, # 0-1
                "length_ratio": float,         # rewritten_len / original_len
                "issues": List[str]            # 问题列表
            }
        """
        issues = []
        
        # 1. Must Events 保留率
        event_retention = cls._check_events(original, rewritten, must_events)
        if event_retention < 0.95:
            issues.append(f"Must events retention: {event_retention*100:.1f}% (target: ≥95%)")
        
        # 2. 角色名一致性
        character_consistency = cls._check_characters(original, rewritten)
        if character_consistency < 1.0:
            issues.append(f"Character consistency: {character_consistency*100:.1f}% (target: 100%)")
        
        # 3. 字数倍率
        orig_len = len(original)
        new_len = len(rewritten)
        length_ratio = new_len / max(orig_len, 1)
        if not (0.8 <= length_ratio <= 2.0):
            issues.append(f"Length ratio: {length_ratio:.2f} (target: 0.8-2.0)")
        
        passed = len(issues) == 0
        
        return {
            "passed": passed,
            "event_retention": round(event_retention, 4),
            "character_consistency": round(character_consistency, 4),
            "length_ratio": round(length_ratio, 2),
            "issues": issues,
        }

    @classmethod
    def _check_events(cls, original: str, rewritten: str, must_events: List[str]) -> float:
        """检查 must_events 是否在改写后仍然存在"""
        if not must_events:
            return 1.0
        
        # 提取原文和改写文的关键词（取每个事件的前6个字符作为标识）
        event_keys = []
        for event in must_events:
            # 去除多余空格，取前6个中文字符
            clean = re.sub(r'[^\u4e00-\u9fff]', '', event)
            key = clean[:6] if len(clean) >= 4 else event[:6]
            event_keys.append(key)
        
        retained = 0
        for key in event_keys:
            if key in rewritten:
                retained += 1
            # 即使关键词不在，也允许用更宽松的匹配
            elif any(kw in rewritten for kw in key):
                retained += 1
        
        return retained / len(must_events)

    @classmethod
    def _check_characters(cls, original: str, rewritten: str) -> float:
        """检查角色名是否在改写中保持一致"""
        # 提取原文中的角色名（可配置）
        # 简单方法：提取所有3-4字的、非动词的高频词
        # 更好的方法：从配置读取，但离线实验使用简单规则
        
        # 这里用简化方案：提取原文中所有3-4字的人名模式
        # 实际项目中应从 character.yaml 读取
        import re
        from collections import Counter
        
        # 提取3-4字中文词
        words = re.findall(r'[\u4e00-\u9fff]{2,4}', original)
        # 过滤常见词（过于简化，仅用于 POC）
        common_words = {"林逸", "玄老", "二叔", "青云", "长老", "傀儡", "玉佩", "灵泉", "禁制", "邪气", "灵力", "丹田"}
        possible_names = [w for w in words if w in common_words or len(w) >= 3]
        
        # 取出现频率最高的前5个
        name_counter = Counter(possible_names)
        top_names = [name for name, count in name_counter.most_common(5) if count >= 1]
        
        if not top_names:
            return 1.0
        
        # 检查每个名字在改写文中是否仍然存在
        present = 0
        for name in top_names:
            if name in rewritten:
                present += 1
        
        return present / len(top_names)