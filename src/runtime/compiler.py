# src/runtime/compiler.py
"""
Compiler v2 - Policy → LayerControlTargets (IR)

职责：将 PolicyConfig 翻译为各传播层（L1-L4）的控制目标。
Compiler 不生成 Prompt，只输出 Runtime 的中间表示（IR）。

这是 Runtime 控制语义的 SSOT（Single Source of Truth）。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any

from src.runtime.models import PolicyConfig, PredictionMode, RealizationMode


class LayerTarget(Enum):
    """各传播层的控制目标"""
    FIXED = "fixed"          # State 不得改变该层
    ASSIST = "assist"        # State 可辅助该层
    PRIMARY = "primary"      # State 主导该层
    ENHANCED = "enhanced"    # State 增强该层
    NORMAL = "normal"        # State 正常注入该层
    NONE = "none"            # State 不涉及该层


@dataclass(frozen=True)
class LayerControlTargets:
    """
    各传播层的控制目标
    
    这是 Runtime 的控制语义中间表示（IR）。
    独立于任何渲染方式（Prompt / XML / JSON）。
    """
    prediction: LayerTarget
    reasoning: LayerTarget
    justification: LayerTarget
    construction: LayerTarget
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "prediction": self.prediction.value,
            "reasoning": self.reasoning.value,
            "justification": self.justification.value,
            "construction": self.construction.value,
        }
    
    @classmethod
    def conservative(cls) -> "LayerControlTargets":
        """Conservative Policy: L1 固定，L2-L4 增强"""
        return cls(
            prediction=LayerTarget.FIXED,
            reasoning=LayerTarget.ENHANCED,
            justification=LayerTarget.ENHANCED,
            construction=LayerTarget.ENHANCED,
        )
    
    @classmethod
    def adaptive(cls) -> "LayerControlTargets":
        """Adaptive Policy: L1 辅助，L2-L4 增强"""
        return cls(
            prediction=LayerTarget.ASSIST,
            reasoning=LayerTarget.ENHANCED,
            justification=LayerTarget.ENHANCED,
            construction=LayerTarget.ENHANCED,
        )
    
    @classmethod
    def aggressive(cls) -> "LayerControlTargets":
        """Aggressive Policy: L1 主导，L2-L4 增强"""
        return cls(
            prediction=LayerTarget.PRIMARY,
            reasoning=LayerTarget.ENHANCED,
            justification=LayerTarget.ENHANCED,
            construction=LayerTarget.ENHANCED,
        )
    
    @classmethod
    def from_policy(cls, policy: PolicyConfig) -> "LayerControlTargets":
        """根据 Policy 计算 LayerTargets"""
        
        # Prediction 目标
        if policy.prediction == PredictionMode.DISABLED:
            prediction = LayerTarget.FIXED
        elif policy.prediction == PredictionMode.ASSIST:
            prediction = LayerTarget.ASSIST
        elif policy.prediction == PredictionMode.PRIMARY:
            prediction = LayerTarget.PRIMARY
        else:
            prediction = LayerTarget.NORMAL
        
        # Reasoning / Justification / Construction 目标
        if policy.realization == RealizationMode.NONE:
            reasoning = LayerTarget.NONE
            justification = LayerTarget.NONE
            construction = LayerTarget.NONE
        elif policy.realization == RealizationMode.NORMAL:
            reasoning = LayerTarget.NORMAL
            justification = LayerTarget.NORMAL
            construction = LayerTarget.NORMAL
        elif policy.realization == RealizationMode.ENHANCED:
            reasoning = LayerTarget.ENHANCED
            justification = LayerTarget.ENHANCED
            construction = LayerTarget.ENHANCED
        else:
            reasoning = LayerTarget.NORMAL
            justification = LayerTarget.NORMAL
            construction = LayerTarget.NORMAL
        
        return cls(
            prediction=prediction,
            reasoning=reasoning,
            justification=justification,
            construction=construction,
        )


class Compiler:
    """
    Compiler v2 - 语义编译
    
    输入：PolicyConfig
    输出：LayerControlTargets (IR)
    
    Version: 2.0
    """
    
    VERSION = "2.0"
    
    def compile(self, policy: PolicyConfig) -> LayerControlTargets:
        """编译 Policy 为 LayerControlTargets"""
        return LayerControlTargets.from_policy(policy)


def compile_policy(policy: PolicyConfig) -> LayerControlTargets:
    """便捷函数"""
    return Compiler().compile(policy)