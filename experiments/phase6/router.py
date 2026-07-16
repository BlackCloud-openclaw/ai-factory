# src/runtime/router.py
"""
Runtime Router - 确定性纯函数策略选择器

职责：
1. 接收 SceneAnalysis（场景特征）
2. 基于 TR 和 Prediction Plasticity 计算各策略的兼容性分数
3. 输出完整的 RouterDecision（含决策轨迹、置信度、Margin、理由）

设计原则：
- 纯函数：相同输入永远相同输出
- 不读取 Narrative、不依赖 Writer、不访问数据库
- 输出完整的决策轨迹，为 Explainability 提供基础
"""

from typing import Tuple
import math

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


class RuntimeRouter:
    """
    确定性 Router - 基于 TR 和场景特征选择传播策略

    路由逻辑（基于 Phase 5 理论和 Phase 6.1 实验结果）：

    TR 区间与策略映射：
    - TR < 0.40:  Highly Open   → Adaptive (PRIMARY + ENHANCED)
    - 0.40 ≤ TR < 0.65: Competitive → Adaptive (ASSIST + ENHANCED)
    - 0.65 ≤ TR < 0.85: Moderately Rigid → Conservative (DISABLED + ENHANCED)
    - TR ≥ 0.85:  Rigid        → Conservative (DISABLED + ENHANCED)
    """

    # Phase 5 经验阈值（冻结）
    THRESHOLD_OPEN = 0.40
    THRESHOLD_COMPETITIVE = 0.65
    THRESHOLD_MODERATELY_RIGID = 0.85

    # 各策略的基础评分（按 TR 区间调整）
    # 分数范围 0-1

    def route(self, analysis: SceneAnalysis) -> RouterDecision:
        """
        确定性路由决策

        Args:
            analysis: 场景分析结果（含 TR、Prediction Plasticity、置信度等）

        Returns:
            RouterDecision: 完整的决策轨迹
        """
        tr = analysis.tr
        confidence = analysis.confidence
        source = analysis.source

        # 1. 计算各策略的兼容性分数
        scores = self._compute_candidate_scores(tr, confidence, source)

        # 2. 确定选中的策略
        selected = self._select_policy(scores)

        # 3. 生成 Policy Config
        policy_config = self._policy_config_for(selected)

        # 4. 计算置信度和 Margin
        confidence_score = self._get_confidence(scores, selected)
        margin = self._compute_margin(scores, selected)

        # 5. 生成决策理由
        rationale = self._generate_rationale(
            tr=tr,
            selected=selected,
            scores=scores,
            confidence=confidence,
            source=source,
        )

        return RouterDecision(
            selected_policy=selected,
            candidate_scores=scores,
            policy_config=policy_config,
            confidence=confidence_score,
            margin=margin,
            rationale=rationale,
            raw_analysis={
                "tr": tr,
                "confidence": confidence,
                "source": source.value,
            },
        )

    # ============================================================
    # 核心决策逻辑
    # ============================================================

    def _compute_candidate_scores(
        self,
        tr: float,
        confidence: float,
        source: AnalysisSource,
    ) -> Tuple[CandidateScore, ...]:
        """
        计算三个策略的兼容性分数

        评分逻辑：
        - Conservative: 基于 Realization ENHANCED，在所有 TR 下都稳定
        - Adaptive: 在低 TR 下最优，中 TR 下良好，高 TR 下下降
        - Aggressive: 仅在低 TR 下有效，高 TR 下会与场景冲突
        """
        # 基础分数（TR 与策略的匹配度）
        conservative_score = self._score_conservative(tr)
        adaptive_score = self._score_adaptive(tr)
        aggressive_score = self._score_aggressive(tr)

        # 置信度加权：低置信度时降低所有分数，趋于保守
        confidence_weight = 0.5 + 0.5 * confidence
        conservative_score *= confidence_weight
        adaptive_score *= confidence_weight
        aggressive_score *= confidence_weight

        # 测量值来源的加分（Phase 5.4 已验证）
        if source == AnalysisSource.MEASURED:
            conservative_score = min(1.0, conservative_score + 0.05)
            adaptive_score = min(1.0, adaptive_score + 0.05)

        return (
            CandidateScore(
                policy=PolicyType.CONSERVATIVE,
                score=conservative_score,
                blocked=False,
            ),
            CandidateScore(
                policy=PolicyType.ADAPTIVE,
                score=adaptive_score,
                blocked=False,
            ),
            CandidateScore(
                policy=PolicyType.AGGRESSIVE,
                score=aggressive_score,
                blocked=aggressive_score < 0.30,  # 极低分时标记为 blocked
            ),
        )

    def _score_conservative(self, tr: float) -> float:
        """
        Conservative 策略（DISABLED + ENHANCED）

        特点：不改变 Prediction，专注 Realization
        - 在所有 TR 下都稳定，高 TR 时最优
        - 低 TR 时仍可用，但不如 Adaptive
        """
        # 基础分：0.70
        # 高 TR (≥0.85) 时 +0.20 → 0.90
        # 中 TR (0.65-0.85) 时 +0.10 → 0.80
        # 低 TR (<0.40) 时 0.70
        if tr >= self.THRESHOLD_MODERATELY_RIGID:
            return 0.90
        elif tr >= self.THRESHOLD_COMPETITIVE:
            return 0.80
        elif tr < self.THRESHOLD_OPEN:
            return 0.70
        else:
            return 0.75

    def _score_adaptive(self, tr: float) -> float:
        """
        Adaptive 策略（ASSIST + ENHANCED）

        特点：同时影响 Prediction 和 Realization
        - 低 TR 时最优（Prediction 有空间）
        - 中 TR 时良好
        - 高 TR 时下降（Prediction 空间有限）
        """
        # 基础分：0.70
        # 低 TR (<0.40) 时 +0.25 → 0.95
        # 中 TR (0.40-0.65) 时 +0.15 → 0.85
        # 高 TR (≥0.85) 时 -0.10 → 0.60
        if tr < self.THRESHOLD_OPEN:
            return 0.95
        elif tr < self.THRESHOLD_COMPETITIVE:
            return 0.85
        elif tr >= self.THRESHOLD_MODERATELY_RIGID:
            return 0.60
        else:
            return 0.75

    def _score_aggressive(self, tr: float) -> float:
        """
        Aggressive 策略（PRIMARY + ENHANCED）

        特点：强制 State 主导 Prediction
        - 仅低 TR 时有效（Prediction 有足够空间）
        - 中 TR 时开始冲突
        - 高 TR 时几乎无效
        """
        # 基础分：0.50
        # 低 TR (<0.40) 时 +0.30 → 0.80
        # 中 TR (0.40-0.65) 时 +0.00 → 0.50
        # 高 TR (≥0.85) 时 -0.40 → 0.10
        if tr < self.THRESHOLD_OPEN:
            return 0.80
        elif tr < self.THRESHOLD_COMPETITIVE:
            return 0.50
        elif tr >= self.THRESHOLD_MODERATELY_RIGID:
            return 0.10
        else:
            return 0.35

    def _select_policy(self, scores: Tuple[CandidateScore, ...]) -> PolicyType:
        """选择分数最高的策略（跳过 blocked 的策略）"""
        valid = [s for s in scores if not s.blocked]
        if not valid:
            # 如果所有都被 blocked，选择 Conservative（最安全）
            return PolicyType.CONSERVATIVE
        return max(valid, key=lambda s: s.score).policy

    def _policy_config_for(self, policy: PolicyType) -> PolicyConfig:
        """根据策略类型生成 PolicyConfig"""
        mapping = {
            PolicyType.CONSERVATIVE: PolicyConfig(
                prediction=PredictionMode.DISABLED,
                realization=RealizationMode.ENHANCED,
                policy_type=PolicyType.CONSERVATIVE,
            ),
            PolicyType.ADAPTIVE: PolicyConfig(
                prediction=PredictionMode.ASSIST,
                realization=RealizationMode.ENHANCED,
                policy_type=PolicyType.ADAPTIVE,
            ),
            PolicyType.AGGRESSIVE: PolicyConfig(
                prediction=PredictionMode.PRIMARY,
                realization=RealizationMode.ENHANCED,
                policy_type=PolicyType.AGGRESSIVE,
            ),
        }
        return mapping[policy]

    def _get_confidence(self, scores: Tuple[CandidateScore, ...], selected: PolicyType) -> float:
        """获取选中策略的置信度（最高分）"""
        for s in scores:
            if s.policy == selected:
                return s.score
        return 0.5

    def _compute_margin(self, scores: Tuple[CandidateScore, ...], selected: PolicyType) -> float:
        """计算最高分与次高分的差值"""
        sorted_scores = sorted(scores, key=lambda s: s.score, reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[0].policy == selected:
            return sorted_scores[0].score - sorted_scores[1].score
        return 0.0

    def _generate_rationale(
        self,
        tr: float,
        selected: PolicyType,
        scores: Tuple[CandidateScore, ...],
        confidence: float,
        source: AnalysisSource,
    ) -> str:
        """生成人类可读的决策理由"""
        # 获取各策略分数
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

        # 添加分数详情
        lines.append(f"策略分数: Conservative={score_map.get(PolicyType.CONSERVATIVE, 0):.2f}, "
                     f"Adaptive={score_map.get(PolicyType.ADAPTIVE, 0):.2f}, "
                     f"Aggressive={score_map.get(PolicyType.AGGRESSIVE, 0):.2f}")

        return "; ".join(lines)


# ============================================================
# 便捷函数
# ============================================================

def route_scene(analysis: SceneAnalysis) -> RouterDecision:
    """便捷函数：快速路由"""
    router = RuntimeRouter()
    return router.route(analysis)


# ============================================================
# 导出
# ============================================================

__all__ = [
    "RuntimeRouter",
    "route_scene",
]