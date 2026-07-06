"""
Narrative KPI Engine - 主入口
Phase 5 核心模块，聚合所有子模块计算 8 个维度和综合分数
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import math

from .narrative_metrics.constants import ENGAGEMENT_DIMS, PROGRESSION_DIMS
from .narrative_metrics.text_extractor import TextExtractor
from .narrative_metrics.state_diff import StateDiffAnalyzer, StateDiffResult
from .narrative_metrics.reader_engagement import ReaderEngagementScorer
from .narrative_metrics.story_progression import StoryProgressionScorer


@dataclass
class NarrativeKPIResult:
    """KPI 计算结果"""
    # 8 个维度分数
    dialogue: float = 0.0
    interaction: float = 0.0
    conflict: float = 0.0
    pressure: float = 0.0
    tension: float = 0.0
    relationship: float = 0.0
    goal: float = 0.0
    character: float = 0.0

    # 聚合分数
    engagement: float = 0.0
    progression: float = 0.0
    narrative_value: float = 0.0

    # 元数据
    total_chars: int = 0
    total_events: int = 0
    versions: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dialogue": self.dialogue,
            "interaction": self.interaction,
            "conflict": self.conflict,
            "pressure": self.pressure,
            "tension": self.tension,
            "relationship": self.relationship,
            "goal": self.goal,
            "character": self.character,
            "engagement": self.engagement,
            "progression": self.progression,
            "narrative_value": self.narrative_value,
            "total_chars": self.total_chars,
            "total_events": self.total_events,
            "versions": self.versions,
        }


class NarrativeKPIEngine:
    """叙事 KPI 计算引擎"""

    def __init__(self, character_names: Optional[List[str]] = None):
        """
        Args:
            character_names: 场景中出现的角色名列表，用于互动检测
        """
        self.character_names = character_names or ["林逸", "苏清雪", "二叔", "玄老", "萧寒"]

    def compute(
        self,
        scene_text: str,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        events: Optional[List[Dict[str, Any]]] = None,
    ) -> NarrativeKPIResult:
        """
        计算场景的完整 KPI

        Args:
            scene_text: 场景正文
            state_before: 场景前的 WorldState (dict)
            state_after: 场景后的 WorldState (dict)
            events: 场景中的事件列表（可选，用于增强特征）

        Returns:
            NarrativeKPIResult
        """
        total_chars = len(scene_text)
        total_events = len(events) if events else 0

        # ----- 1. 计算状态差异 -----
        state_diff = StateDiffAnalyzer.analyze(state_before, state_after)

        # ----- 2. 计算 Reader Engagement -----
        engagement_scores = ReaderEngagementScorer.score_all(
            text=scene_text,
            character_names=self.character_names,
            state_diff_score=state_diff.relationship_delta,
        )

        # ----- 3. 计算 Story Progression -----
        progression_scores = StoryProgressionScorer.score_all(
            text=scene_text,
            state_diff=state_diff,
        )

        # ----- 4. 聚合 -----
        engagement = sum(engagement_scores.values()) / len(ENGAGEMENT_DIMS)
        progression = sum(progression_scores.values()) / len(PROGRESSION_DIMS)

        # 核心公式：√(Engagement × Progression)
        narrative_value = math.sqrt(engagement * progression)

        # ----- 5. 组装结果 -----
        return NarrativeKPIResult(
            dialogue=engagement_scores["dialogue"],
            interaction=engagement_scores["interaction"],
            conflict=engagement_scores["conflict"],
            pressure=engagement_scores["pressure"],
            tension=engagement_scores["tension"],
            relationship=progression_scores["relationship"],
            goal=progression_scores["goal"],
            character=progression_scores["character"],
            engagement=engagement,
            progression=progression,
            narrative_value=narrative_value,
            total_chars=total_chars,
            total_events=total_events,
            versions={
                "engine": "v1.0",
                "spec": "narrative_kpi_spec_v1",
            },
        )