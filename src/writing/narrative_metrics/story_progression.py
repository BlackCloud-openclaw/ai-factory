"""
Story Progression 评分器
计算 3 个 Progression 维度：Relationship, Goal, Character
"""

from typing import Dict, Any, List

from .constants import PROGRESSION_DIMS
from .text_extractor import TextExtractor
from .state_diff import StateDiffResult


class StoryProgressionScorer:
    """故事推进评分器"""

    @staticmethod
    def score_relationship(state_diff: StateDiffResult) -> float:
        """
        Relationship Movement（1-5）
        基于：关系变化量 + 质变标志
        """
        delta = state_diff.relationship_delta
        change_count = state_diff.relationship_change_count

        # 基础分
        if delta < 0.05 or change_count == 0:
            base = 1.0
        elif delta < 0.15:
            base = 2.0
        elif delta < 0.3:
            base = 3.0
        elif delta < 0.5:
            base = 4.0
        else:
            base = 4.5

        # 质变检测：如果有关系的极性发生变化（+ -> - 或 - -> +）
        # 这里无法从 StateDiffResult 直接获取，通过文本辅助检测
        # 简化：如果有 "决裂"、"和解"、"联盟" 等词
        return min(5.0, max(1.0, round(base * 2) / 2))

    @staticmethod
    def score_goal(text: str, state_diff: StateDiffResult) -> float:
        """
        Goal Advancement（1-5）
        基于：目标状态是否被重新定义
        """
        goal_changed = state_diff.goal_state_changed

        # 检测文本中的目标重定义关键词
        redef_score = TextExtractor.estimate_goal_redefinition(text)

        # 检测路径改变
        path_changed = any(kw in text for kw in ["改道", "绕路", "转向", "改变计划"])

        # 检测目标替换
        goal_replaced = any(kw in text for kw in ["不再是", "真正目标是", "原来是为了"])

        if not goal_changed and redef_score < 0.2 and not path_changed and not goal_replaced:
            base = 1.0
        elif redef_score < 0.3 and path_changed:
            base = 3.0  # 路径改变
        elif redef_score >= 0.4 or goal_replaced:
            base = 4.5  # 目标替换
        elif goal_changed:
            base = 4.0  # 状态变化
        else:
            base = 2.0

        # 检测是否 "发现关键信息"（推进但非替换）
        if any(kw in text for kw in ["发现", "关键", "重要线索"]):
            if base < 3.0:
                base += 0.5

        return min(5.0, max(1.0, round(base * 2) / 2))

    @staticmethod
    def score_character(text: str, state_diff: StateDiffResult) -> float:
        """
        Character Change（1-5）
        基于：决策模型是否改变（认知 + 行为 + 身份）
        """
        cognitive = state_diff.cognitive_model_changed
        behavior = state_diff.behavior_changed
        identity = state_diff.identity_changed

        # 文本特征检测
        cognitive_kw = TextExtractor.detect_cognitive_shift(text)
        behavior_kw = TextExtractor.detect_behavior_change(text)
        identity_kw = TextExtractor.detect_identity_shift(text)

        # 综合判断
        has_cognitive = cognitive or cognitive_kw > 0
        has_behavior = behavior or behavior_kw > 0
        has_identity = identity or identity_kw > 0

        if not has_cognitive and not has_behavior and not has_identity:
            base = 1.0
        elif has_cognitive and not has_behavior and not has_identity:
            base = 3.0  # 认知模型改变
        elif has_behavior and not has_identity:
            base = 4.0  # 行为准则改变
        elif has_identity:
            base = 5.0  # 身份认同改变
        else:
            base = 2.0  # 仅产生怀疑

        # 强化：如果检测到行为改变关键词
        if behavior_kw > 1 and base < 4.0:
            base = 4.0

        return min(5.0, max(1.0, round(base * 2) / 2))

    @classmethod
    def score_all(
        cls,
        text: str,
        state_diff: StateDiffResult,
    ) -> Dict[str, float]:
        """计算所有 Progression 维度"""
        return {
            "relationship": cls.score_relationship(state_diff),
            "goal": cls.score_goal(text, state_diff),
            "character": cls.score_character(text, state_diff),
        }