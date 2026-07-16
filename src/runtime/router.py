# src/runtime/router.py
"""
Runtime Router - 运行时路由决策

包含：
- RetryStrategy: 重试策略枚举
- PredictionMode, RealizationMode: 已移至 models.py
- RoutingDecision: 路由决策
- RuntimeRouter: 路由器实现
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

from src.runtime.models import (
    SceneAnalysis,
    RouterDecision,
    CandidateScore,
    PolicyConfig,
    PolicyType,
    PredictionMode,
    RealizationMode,
    AnalysisSource,
)

logger = logging.getLogger(__name__)


# ============================================================
# 重试策略（与 models 中的其他枚举协调）
# ============================================================

class RetryStrategy(Enum):
    """验证失败时的重试策略"""
    FULL = "full"
    REALIZATION_ONLY = "realization_only"
    NONE = "none"


# ============================================================
# 路由决策（与 models 中的定义一致）
# ============================================================

@dataclass
class RoutingDecision:
    """Runtime Router 的旧版输出（保留兼容）"""
    prediction: PredictionMode
    realization: RealizationMode
    retry: RetryStrategy
    confidence: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "prediction": self.prediction.value,
            "realization": self.realization.value,
            "retry": self.retry.value,
            "confidence": self.confidence,
            "reason": self.reason,
        }

    def is_prediction_enabled(self) -> bool:
        return self.prediction in (PredictionMode.ASSIST, PredictionMode.PRIMARY)

    def is_realization_enabled(self) -> bool:
        return self.realization in (RealizationMode.NORMAL, RealizationMode.ENHANCED)

    def should_retry(self) -> bool:
        return self.retry != RetryStrategy.NONE


# ============================================================
# 运行时路由器
# ============================================================

class RuntimeRouter:
    THRESHOLD_OPEN = 0.40
    THRESHOLD_COMPETITIVE = 0.65
    THRESHOLD_MODERATELY_RIGID = 0.85

    def route(self, analysis: SceneAnalysis) -> RouterDecision:
        tr = analysis.tr
        confidence = analysis.confidence
        source = analysis.source

        scores = self._compute_candidate_scores(tr, confidence, source)
        selected = self._select_policy(scores)
        policy_config = self._policy_config_for(selected)
        confidence_score = self._get_confidence(scores, selected)
        margin = self._compute_margin(scores, selected)
        rationale = self._generate_rationale(tr, selected, scores, confidence, source)

        return RouterDecision(
            selected_policy=selected,
            candidate_scores=scores,
            policy_config=policy_config,
            confidence=confidence_score,
            margin=margin,
            rationale=rationale,
            raw_analysis={"tr": tr, "confidence": confidence, "source": source.value},
        )

    def _compute_candidate_scores(self, tr: float, confidence: float, source: AnalysisSource) -> Tuple[CandidateScore, ...]:
        conservative = self._score_conservative(tr)
        adaptive = self._score_adaptive(tr)
        aggressive = self._score_aggressive(tr)

        weight = 0.5 + 0.5 * confidence
        conservative *= weight
        adaptive *= weight
        aggressive *= weight

        if source == AnalysisSource.MEASURED:
            conservative = min(1.0, conservative + 0.05)
            adaptive = min(1.0, adaptive + 0.05)

        return (
            CandidateScore(PolicyType.CONSERVATIVE, conservative, blocked=False),
            CandidateScore(PolicyType.ADAPTIVE, adaptive, blocked=False),
            CandidateScore(PolicyType.AGGRESSIVE, aggressive, blocked=aggressive < 0.30),
        )

    def _score_conservative(self, tr: float) -> float:
        if tr >= self.THRESHOLD_MODERATELY_RIGID:
            return 0.90
        elif tr >= self.THRESHOLD_COMPETITIVE:
            return 0.80
        elif tr < self.THRESHOLD_OPEN:
            return 0.70
        else:
            return 0.75

    def _score_adaptive(self, tr: float) -> float:
        if tr < self.THRESHOLD_OPEN:
            return 0.95
        elif tr < self.THRESHOLD_COMPETITIVE:
            return 0.85
        elif tr >= self.THRESHOLD_MODERATELY_RIGID:
            return 0.60
        else:
            return 0.75

    def _score_aggressive(self, tr: float) -> float:
        if tr < self.THRESHOLD_OPEN:
            return 0.80
        elif tr < self.THRESHOLD_COMPETITIVE:
            return 0.50
        elif tr >= self.THRESHOLD_MODERATELY_RIGID:
            return 0.10
        else:
            return 0.35

    def _select_policy(self, scores: Tuple[CandidateScore, ...]) -> PolicyType:
        valid = [s for s in scores if not s.blocked]
        if not valid:
            return PolicyType.CONSERVATIVE
        return max(valid, key=lambda s: s.score).policy

    def _policy_config_for(self, policy: PolicyType) -> PolicyConfig:
        mapping = {
            PolicyType.CONSERVATIVE: PolicyConfig(PredictionMode.DISABLED, RealizationMode.ENHANCED, PolicyType.CONSERVATIVE),
            PolicyType.ADAPTIVE: PolicyConfig(PredictionMode.ASSIST, RealizationMode.ENHANCED, PolicyType.ADAPTIVE),
            PolicyType.AGGRESSIVE: PolicyConfig(PredictionMode.PRIMARY, RealizationMode.ENHANCED, PolicyType.AGGRESSIVE),
        }
        return mapping[policy]

    def _get_confidence(self, scores: Tuple[CandidateScore, ...], selected: PolicyType) -> float:
        for s in scores:
            if s.policy == selected:
                return s.score
        return 0.5

    def _compute_margin(self, scores: Tuple[CandidateScore, ...], selected: PolicyType) -> float:
        sorted_scores = sorted(scores, key=lambda s: s.score, reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[0].policy == selected:
            return sorted_scores[0].score - sorted_scores[1].score
        return 0.0

    def _generate_rationale(self, tr: float, selected: PolicyType, scores: Tuple[CandidateScore, ...], confidence: float, source: AnalysisSource) -> str:
        score_map = {s.policy: s.score for s in scores}
        lines = []
        lines.append(f"TR={tr:.2f}, source={source.value}")
        if source == AnalysisSource.MEASURED:
            lines.append("TR 来自实验测量值（置信度高）")
        if selected == PolicyType.CONSERVATIVE:
            if tr >= self.THRESHOLD_MODERATELY_RIGID:
                lines.append("TR 较高（≥0.85），Prediction 空间有限，选择保守策略")
            elif tr >= self.THRESHOLD_COMPETITIVE:
                lines.append("TR 中等（0.65-0.85），Prediction 有一定空间，保守策略更安全")
            else:
                lines.append("TR 较低，保守策略仍稳定可用")
        elif selected == PolicyType.ADAPTIVE:
            if tr < self.THRESHOLD_OPEN:
                lines.append("TR 低（<0.40），Prediction 空间充足，选择自适应策略")
            elif tr < self.THRESHOLD_COMPETITIVE:
                lines.append("TR 中等（0.40-0.65），自适应策略可在 Prediction 和 Realization 间平衡")
            else:
                lines.append("TR 偏高，自适应策略仍有一定效果但非最优")
        elif selected == PolicyType.AGGRESSIVE:
            if tr < self.THRESHOLD_OPEN:
                lines.append("TR 低（<0.40），激进策略可最大化 State 主导预测的效果")
            else:
                lines.append("TR 偏高，激进策略可能存在兼容性风险")
        lines.append(f"策略分数: Conservative={score_map.get(PolicyType.CONSERVATIVE, 0):.2f}, "
                     f"Adaptive={score_map.get(PolicyType.ADAPTIVE, 0):.2f}, "
                     f"Aggressive={score_map.get(PolicyType.AGGRESSIVE, 0):.2f}")
        return "; ".join(lines)


def route_scene(analysis: SceneAnalysis) -> RouterDecision:
    return RuntimeRouter().route(analysis)