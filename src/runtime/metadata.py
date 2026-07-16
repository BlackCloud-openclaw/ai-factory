from __future__ import annotations

"""
Runtime Metadata - 运行时元数据聚合

职责：
1. 聚合 SceneAnalysis 和 RoutingDecision
2. 提供统一的运行时元数据接口
3. 支持序列化/反序列化

这是 Runtime Metadata Pipeline 的第三步。
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging

from src.runtime.scene_analyzer import SceneAnalysis, SceneAnalyzer, AnalysisSource
from src.runtime.router import (
    RoutingDecision,
    PredictionMode,
    RealizationMode,
    RetryStrategy,
    RuntimeRouter,
)

logger = logging.getLogger(__name__)


@dataclass
class RuntimeMetadata:
    """
    运行时元数据
    
    包含场景分析和路由决策，是 Writer 接收的唯一元数据对象。
    """
    analysis: SceneAnalysis
    decision: RoutingDecision
    # 未来扩展（预留）：
    # emotion: Optional[EmotionState] = None
    # pacing: Optional[PacingState] = None
    
    # ============================================================
    # 便捷属性（用于 Writer 直接访问）
    # ============================================================
    
    @property
    def transition_rigidity(self) -> float:
        """场景的 Transition Rigidity (TR) 值"""
        return self.analysis.transition_rigidity
    
    @property
    def confidence(self) -> float:
        """TR 的置信度"""
        return self.analysis.confidence
    
    @property
    def prediction(self) -> PredictionMode:
        """Prediction Layer 的策略"""
        return self.decision.prediction
    
    @property
    def realization(self) -> RealizationMode:
        """Realization Layer 的策略"""
        return self.decision.realization
    
    @property
    def retry(self) -> RetryStrategy:
        """重试策略"""
        return self.decision.retry
    
    # ============================================================
    # 序列化
    # ============================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "analysis": self.analysis.to_dict(),
            "decision": self.decision.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RuntimeMetadata:
        """从字典反序列化"""
        from src.runtime.scene_analyzer import SceneAnalysis
        from src.runtime.router import RoutingDecision
        
        analysis_data = data.get("analysis", {})
        decision_data = data.get("decision", {})
        
        # 重建 SceneAnalysis
        analysis = SceneAnalysis(
            transition_rigidity=analysis_data.get("transition_rigidity", 0.5),
            confidence=analysis_data.get("confidence", 0.5),
            source=analysis_data.get("source", "default"),
            reason=analysis_data.get("reason", ""),
            features=analysis_data.get("features", {}),
        )
        
        # 重建 RoutingDecision
        decision = RoutingDecision(
            prediction=PredictionMode(decision_data.get("prediction", "assist")),
            realization=RealizationMode(decision_data.get("realization", "normal")),
            retry=RetryStrategy(decision_data.get("retry", "none")),
            confidence=decision_data.get("confidence", 0.5),
            reason=decision_data.get("reason", ""),
        )
        
        return cls(analysis=analysis, decision=decision)
    
    # ============================================================
    # 工厂方法
    # ============================================================
    
    @classmethod
    def from_scene_plan(
        cls,
        scene_plan: Dict[str, Any],
        analyzer: Optional[SceneAnalyzer] = None,
        router: Optional[RuntimeRouter] = None,
    ) -> RuntimeMetadata:
        """
        从场景计划构建 RuntimeMetadata
        
        这是最常用的入口方法。
        
        Args:
            scene_plan: 场景计划
            analyzer: 可选的 SceneAnalyzer 实例
            router: 可选的 RuntimeRouter 实例
            
        Returns:
            RuntimeMetadata
        """
        analyzer = analyzer or SceneAnalyzer()
        router = router or RuntimeRouter()
        
        analysis = analyzer.analyze(scene_plan)
        decision = router.route(analysis)
        
        logger.debug(
            f"RuntimeMetadata built: TR={analysis.transition_rigidity:.2f}, "
            f"prediction={decision.prediction.value}, "
            f"realization={decision.realization.value}"
        )
        
        return cls(analysis=analysis, decision=decision)
    
    # ============================================================
    # 便捷判断方法
    # ============================================================
    
    def is_prediction_enabled(self) -> bool:
        """是否允许改变 Prediction Layer"""
        return self.decision.is_prediction_enabled()
    
    def is_realization_enabled(self) -> bool:
        """是否允许在 Realization Layer 注入 State"""
        return self.decision.is_realization_enabled()
    
    def should_retry(self) -> bool:
        """是否需要重试"""
        return self.decision.should_retry()
    
    def get_summary(self) -> str:
        """获取人类可读的摘要"""
        return (
            f"TR={self.transition_rigidity:.2f} "
            f"(conf={self.confidence:.2f}) → "
            f"pred={self.prediction.value}, "
            f"real={self.realization.value}"
        )


# ============================================================
# 模块级便捷函数
# ============================================================

def build_runtime_metadata(scene_plan: Dict[str, Any]) -> RuntimeMetadata:
    """
    便捷函数：快速构建 RuntimeMetadata
    
    这是最常用的外部入口。
    
    Args:
        scene_plan: 场景计划
        
    Returns:
        RuntimeMetadata
    """
    return RuntimeMetadata.from_scene_plan(scene_plan)