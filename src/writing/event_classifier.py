# src/writing/event_classifier.py
"""
Event Classifier - Phase 13.2.3A

将 must_event 文本分类为 EventType。
确定性实现，仅基于关键词匹配，支持多事件分类。
"""

from enum import Enum
from typing import Optional, Set, List
import re


class EventType(str, Enum):
    """叙事事件类型 - 与 World State 变更直接相关。"""
    REALM_ADVANCE = "realm_advance"       # 境界突破
    ITEM_ACQUIRE = "item_acquire"         # 获得物品
    ITEM_LOST = "item_lost"               # 失去物品
    LOCATION_CHANGE = "location_change"   # 地点变化
    RELATION_CHANGE = "relation_change"   # 关系变化
    PLOT_REVEAL = "plot_reveal"           # 关键信息揭示 (plot_flag)


class EventClassifier:
    """
    确定性事件分类器。

    规则：
    - 基于关键词匹配
    - 返回所有匹配的事件类型（一个事件文本可能产生多个 StateChange）
    - 如果无法分类，返回空列表
    """

    # 关键词映射（优先级由列表顺序决定：越靠前优先级越高）
    KEYWORD_RULES = [
        # Realm Advance (高优先级，避免与 location 混淆)
        ({"突破", "晋升", "晋级", "渡劫", "破境", "升阶", "进阶"}, EventType.REALM_ADVANCE),

        # Item Acquire
        ({"获得", "捡到", "得到", "夺取", "缴获", "拾取", "入手", "拿到"}, EventType.ITEM_ACQUIRE),

        # Item Lost
        ({"失去", "丢失", "遗失", "被夺", "被抢", "销毁"}, EventType.ITEM_LOST),

        # Location Change
        ({"进入", "踏入", "抵达", "来到", "返回", "离开", "走出"}, EventType.LOCATION_CHANGE),

        # Relation Change
        ({"关系", "交恶", "结盟", "决裂", "和解", "结仇", "结怨", "亲密度"}, EventType.RELATION_CHANGE),

        # Plot Reveal
        ({"发现", "揭示", "揭晓", "暴露", "真相", "秘密", "线索", "查明"}, EventType.PLOT_REVEAL),
    ]

    # 停用词（避免无意义匹配）
    STOP_WORDS = {"的", "了", "是", "在", "和", "与", "或", "但", "而", "被", "把", "让"}

    @classmethod
    def classify(cls, text: str) -> List[EventType]:
        """
        返回所有匹配的事件类型。

        Args:
            text: 事件描述（如 "林逸进入秘境获得神秘玉佩"）

        Returns:
            List[EventType]: 所有匹配的事件类型，去重
        """
        if not text or not text.strip():
            return []

        normalized = cls._normalize(text)
        results = []

        # 按顺序扫描所有规则，不提前退出
        for keywords, event_type in cls.KEYWORD_RULES:
            for kw in keywords:
                if kw in normalized or kw in text:
                    results.append(event_type)
                    break  # 同一规则只匹配一次

        # 去重但保留顺序
        seen = set()
        unique = []
        for et in results:
            if et not in seen:
                seen.add(et)
                unique.append(et)

        return unique

    @classmethod
    def _normalize(cls, text: str) -> str:
        """提取关键词并去除停用词。"""
        # 提取中文字符（2-4 字词）
        words = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
        # 过滤停用词
        filtered = [w for w in words if w not in cls.STOP_WORDS]
        # 去重并保留顺序
        seen = set()
        unique = []
        for w in filtered:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        return "".join(unique)

    @classmethod
    def classify_batch(cls, texts: List[str]) -> List[List[EventType]]:
        """批量分类。"""
        return [cls.classify(t) for t in texts]