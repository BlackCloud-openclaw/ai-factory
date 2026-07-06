"""
Reader Engagement 评分器
计算 5 个 Engagement 维度：Dialogue, Interaction, Conflict, Pressure, Tension
"""

from typing import Dict, Any, List

from .constants import ENGAGEMENT_DIMS
from .text_extractor import TextExtractor


class ReaderEngagementScorer:
    """读者卷入度评分器"""

    @staticmethod
    def score_dialogue(text: str) -> float:
        """
        Dialogue Richness（1-5）
        基于：对话占比 + 说话人数 + 轮次 + 博弈强度
        """
        total_chars = len(text)
        if total_chars < 50:
            return 1.0

        dialogue_chars = TextExtractor.extract_dialogue_chars(text)
        dialogue_ratio = dialogue_chars / total_chars

        speakers = TextExtractor.count_speakers(text)
        turns = TextExtractor.count_dialogue_turns(text)

        # 基础分：对话占比
        if dialogue_ratio < 0.05:
            base = 1.0
        elif dialogue_ratio < 0.15:
            base = 2.0
        elif dialogue_ratio < 0.30:
            base = 3.0
        elif dialogue_ratio < 0.50:
            base = 4.0
        else:
            base = 4.5

        # 调整：说话人多样性
        if speakers >= 3:
            base += 0.5
        elif speakers >= 2:
            base += 0.2

        # 调整：轮次
        if turns >= 3:
            base += 0.5
        elif turns >= 1:
            base += 0.2

        # 调整：检测是否有潜台词（"其实"、"不过"、"你以为"等）
        if any(kw in text for kw in ["其实", "你以为", "不过", "难道", "对吗"]):
            base += 0.3

        return min(5.0, max(1.0, round(base * 2) / 2))  # 四舍五入到 0.5

    @staticmethod
    def score_interaction(text: str, character_names: List[str]) -> float:
        """
        Interaction（1-5）
        基于：角色交替次数 + 行为影响关键词
        """
        if len(character_names) < 2:
            return 1.0

        interactions = TextExtractor.count_character_interactions(text, character_names)

        # 每 3 次交替算 1 次有效互动
        effective_interactions = interactions / 3

        if effective_interactions == 0:
            base = 1.0
        elif effective_interactions < 1:
            base = 2.0
        elif effective_interactions < 2:
            base = 3.0
        elif effective_interactions < 4:
            base = 4.0
        else:
            base = 4.5

        # 检测 "迫使"、"影响"、"改变" 等高影响关键词
        impact_keywords = ["迫使", "影响", "改变", "让", "逼", "迫使", "促使", "导致"]
        if any(kw in text for kw in impact_keywords):
            base += 0.5

        return min(5.0, max(1.0, round(base * 2) / 2))

    @staticmethod
    def score_conflict(text: str, state_diff_score: float = 0.0) -> float:
        """
        Conflict（1-5）
        基于：冲突关键词密度 + 世界状态阻碍
        """
        total_chars = len(text)
        if total_chars < 50:
            return 1.0

        conflict_kw_count = TextExtractor.detect_conflict_keywords(text)
        density = conflict_kw_count / (total_chars / 100)  # 每 100 字符冲突词数

        # 基础分
        if density < 0.5:
            base = 1.0
        elif density < 1.5:
            base = 2.5
        elif density < 3.0:
            base = 3.5
        else:
            base = 4.5

        # 考虑状态差异中的阻碍（如果有）
        if state_diff_score > 0.3:
            base += 0.5

        # 检测内部冲突（"挣扎"、"犹豫"、"两难"）
        internal_keywords = ["挣扎", "犹豫", "两难", "痛苦", "煎熬", "矛盾"]
        if any(kw in text for kw in internal_keywords):
            base += 0.5

        return min(5.0, max(1.0, round(base * 2) / 2))

    @staticmethod
    def score_pressure(text: str) -> float:
        """
        Pressure（1-5）
        基于：压力关键词 + 时间限制 + 代价关键词
        """
        total_chars = len(text)
        if total_chars < 50:
            return 1.0

        pressure_kw_count = TextExtractor.detect_pressure_keywords(text)
        density = pressure_kw_count / (total_chars / 100)

        # 检测时间限制
        time_limited = any(kw in text for kw in ["三日内", "必须", "来不及", "最后", "紧迫"])

        # 检测代价
        cost_high = any(kw in text for kw in ["牺牲", "代价", "换", "不可逆", "失去"])

        # 基础分
        if density < 0.5 and not time_limited and not cost_high:
            base = 1.0
        elif density < 1.0 and not time_limited:
            base = 2.0
        elif density < 2.0 and (time_limited or cost_high):
            base = 3.5
        elif time_limited and cost_high:
            base = 4.5
        else:
            base = 4.0

        # 检测"必须选择"（高压力信号）
        if any(kw in text for kw in ["必须选择", "不得不", "只能", "否则"]):
            base += 0.5

        return min(5.0, max(1.0, round(base * 2) / 2))

    @staticmethod
    def score_tension(text: str) -> float:
        """
        Narrative Tension（1-5）
        基于：结尾悬念密度 + 未解问题数量
        """
        total_chars = len(text)
        if total_chars < 50:
            return 1.0

        # 检测结尾张力关键词
        tension_score = TextExtractor.detect_tension_keywords(text, tail_ratio=0.15)

        # 检测是否以问题结尾
        ends_with_question = text.strip().endswith("？") or text.strip().endswith("?")

        # 检测"未解"模式
        unresolved_patterns = ["未解", "未知", "真相", "秘密", "究竟"]
        unresolved_count = sum(1 for p in unresolved_patterns if p in text)

        if tension_score == 0 and not ends_with_question and unresolved_count == 0:
            base = 1.0
        elif tension_score < 0.3 and unresolved_count <= 1:
            base = 2.0
        elif tension_score < 0.5 or unresolved_count == 2:
            base = 3.5
        elif tension_score >= 0.5 or ends_with_question or unresolved_count >= 3:
            base = 4.5
        else:
            base = 3.0

        # 如果结尾有明显钩子（"然而"、"忽然"、"?")
        if ends_with_question:
            base += 0.5

        return min(5.0, max(1.0, round(base * 2) / 2))

    @classmethod
    def score_all(
        cls,
        text: str,
        character_names: List[str],
        state_diff_score: float = 0.0,
    ) -> Dict[str, float]:
        """计算所有 Engagement 维度"""
        return {
            "dialogue": cls.score_dialogue(text),
            "interaction": cls.score_interaction(text, character_names),
            "conflict": cls.score_conflict(text, state_diff_score),
            "pressure": cls.score_pressure(text),
            "tension": cls.score_tension(text),
        }