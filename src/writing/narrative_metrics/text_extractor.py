"""
文本特征提取器 - 不依赖 LLM，使用规则和正则表达式
"""

import re
from typing import List, Set, Dict, Any, Optional
from collections import Counter

from .constants import (
    CONFLICT_KEYWORDS,
    PRESSURE_KEYWORDS,
    COGNITIVE_SHIFT_KEYWORDS,
    BEHAVIOR_CHANGE_KEYWORDS,
    IDENTITY_SHIFT_KEYWORDS,
    TENSION_KEYWORDS,
    GOAL_REDEFINE_KEYWORDS,
)


class TextExtractor:
    """从纯文本中提取叙事特征"""

    @staticmethod
    def extract_dialogue_blocks(text: str) -> List[str]:
        """提取所有对话块（中文/英文引号）"""
        pattern = r'[「『"“][^」』”"]*[」』”"]'
        matches = re.findall(pattern, text)
        return matches

    @staticmethod
    def extract_dialogue_chars(text: str) -> int:
        """对话总字符数"""
        blocks = TextExtractor.extract_dialogue_blocks(text)
        return sum(len(b) for b in blocks)

    @staticmethod
    def count_speakers(text: str) -> int:
        """统计说话人数量（基于"角色名："或"角色名说"模式）"""
        # 匹配 "角色名：" 或 "角色名说"
        pattern1 = r'([\u4e00-\u9fff]{2,4})[:：]'
        pattern2 = r'([\u4e00-\u9fff]{2,4})(?:说|道|问|答|笑|叹|喊)'
        speakers = set()
        speakers.update(re.findall(pattern1, text))
        speakers.update(re.findall(pattern2, text))
        return len(speakers)

    @staticmethod
    def count_dialogue_turns(text: str) -> int:
        """
        计算对话轮次（A→B 算一次完整轮次）
        粗略估计：对话块数量的一半
        """
        blocks = TextExtractor.extract_dialogue_blocks(text)
        return len(blocks) // 2

    @staticmethod
    def detect_conflict_keywords(text: str) -> int:
        """统计冲突关键词出现次数"""
        count = 0
        for kw in CONFLICT_KEYWORDS:
            count += text.count(kw)
        return count

    @staticmethod
    def detect_pressure_keywords(text: str) -> int:
        """统计压力关键词出现次数"""
        count = 0
        for kw in PRESSURE_KEYWORDS:
            count += text.count(kw)
        return count

    @staticmethod
    def detect_cognitive_shift(text: str) -> int:
        """检测认知转变关键词"""
        count = 0
        for kw in COGNITIVE_SHIFT_KEYWORDS:
            count += text.count(kw)
        return count

    @staticmethod
    def detect_behavior_change(text: str) -> int:
        """检测行为改变关键词"""
        count = 0
        for kw in BEHAVIOR_CHANGE_KEYWORDS:
            count += text.count(kw)
        return count

    @staticmethod
    def detect_identity_shift(text: str) -> int:
        """检测身份转变关键词"""
        count = 0
        for kw in IDENTITY_SHIFT_KEYWORDS:
            count += text.count(kw)
        return count

    @staticmethod
    def detect_tension_keywords(text: str, tail_ratio: float = 0.1) -> float:
        """
        检测结尾部分的张力关键词密度
        重点检测文本末尾 10-20%
        """
        if len(text) < 50:
            return 0.0
        tail_len = max(200, int(len(text) * tail_ratio))
        tail = text[-tail_len:]

        count = 0
        for kw in TENSION_KEYWORDS:
            count += tail.count(kw)

        # 归一化到 0-1
        max_expected = 10
        return min(1.0, count / max_expected)

    @staticmethod
    def detect_goal_redefine(text: str) -> int:
        """检测目标重定义关键词"""
        count = 0
        for kw in GOAL_REDEFINE_KEYWORDS:
            count += text.count(kw)
        return count

    @staticmethod
    def extract_character_mentions(text: str, character_names: List[str]) -> Dict[str, int]:
        """统计每个角色出现的次数"""
        counts = {}
        for name in character_names:
            counts[name] = text.count(name)
        return counts

    @staticmethod
    def count_character_interactions(text: str, character_names: List[str]) -> int:
        """
        检测角色交替模式（互动次数）
        例如：林逸...苏清雪...林逸 → 1 次交替
        """
        if not character_names or len(character_names) < 2:
            return 0

        # 提取所有角色出现的位置
        mentions = []
        for name in character_names:
            for match in re.finditer(name, text):
                mentions.append((match.start(), name))

        mentions.sort(key=lambda x: x[0])

        # 统计交替次数
        interactions = 0
        for i in range(1, len(mentions)):
            if mentions[i][1] != mentions[i-1][1]:
                interactions += 1

        return interactions

    @staticmethod
    def estimate_goal_redefinition(text: str) -> float:
        """
        估算目标是否被重新定义
        返回 0-1 分数
        """
        # 检测关键词
        keyword_score = min(1.0, TextExtractor.detect_goal_redefine(text) / 3.0)

        # 检测 "发现...真正..." 模式
        if re.search(r'发现.*?真正', text):
            keyword_score = max(keyword_score, 0.6)

        # 检测 "原来...是..." 模式
        if re.search(r'原来.*?是', text):
            keyword_score = max(keyword_score, 0.5)

        return keyword_score