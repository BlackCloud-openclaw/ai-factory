"""
锚点提取器 - 从原文中提取不可变元素
"""

import re
from typing import List, Set, Dict, Any


class AnchorExtractor:
    """从文本中提取锚点（角色名、物品名、境界名）"""
    
    # 可配置的锚点库（可从 character.yaml 加载）
    KNOWN_NAMES = {"林逸", "玄老", "二叔", "苏清月", "青云长老", "萧寒", "落魄修士"}
    KNOWN_ITEMS = {"神秘玉佩", "玄天剑诀", "青锋剑", "九幽古篆", "本命法宝", "储物袋"}
    KNOWN_REALMS = {"炼气", "筑基", "金丹", "元婴", "化神", "炼虚", "合体", "大乘", "渡劫"}
    
    @classmethod
    def extract_characters(cls, text: str) -> List[str]:
        """提取文本中出现的所有角色名"""
        found = []
        for name in cls.KNOWN_NAMES:
            if name in text:
                found.append(name)
        return found
    
    @classmethod
    def extract_items(cls, text: str) -> List[str]:
        """提取文本中出现的所有关键物品"""
        found = []
        for item in cls.KNOWN_ITEMS:
            if item in text:
                found.append(item)
        return found
    
    @classmethod
    def extract_realms(cls, text: str) -> List[str]:
        """提取文本中出现的所有境界"""
        found = []
        for realm in cls.KNOWN_REALMS:
            if realm in text:
                found.append(realm)
        return found
    
    @classmethod
    def extract_all(cls, text: str) -> Dict[str, List[str]]:
        """提取所有锚点"""
        return {
            "characters": cls.extract_characters(text),
            "items": cls.extract_items(text),
            "realms": cls.extract_realms(text),
        }