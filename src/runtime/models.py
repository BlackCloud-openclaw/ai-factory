# src/runtime/models.py
"""
Runtime Models - 数据契约（Data Contract）

本模块定义了 Runtime 系统的核心数据结构，是 Runtime 各模块之间的通信协议。

设计原则：
1. 所有 Artifact 数据类均为不可变（frozen=True），确保 SSOT（Single Source of Truth）
2. 枚举类型替代字符串，避免拼写错误，便于 IDE 支持
3. 领域对象存储事实（Fact），派生指标（Score/Gap）由 metrics 模块计算
4. 所有 dataclass 可序列化为 JSON/YAML，用于持久化和调试
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Dict, Any, Optional, List
from datetime import datetime


# ============================================================
# 枚举类型（Enums）
# ============================================================

class PolicyType(Enum):
    """传播策略类型"""
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"
    AGGRESSIVE = "aggressive"


class PredictionMode(Enum):
    """Prediction Layer 的控制模式"""
    DISABLED = "disabled"
    ASSIST = "assist"
    PRIMARY = "primary"


class RealizationMode(Enum):
    """Realization Layer 的控制模式"""
    NONE = "none"
    NORMAL = "normal"
    ENHANCED = "enhanced"


class ReasoningLevel(Enum):
    """L2: State Integration 层级"""
    IGNORED = "ignored"
    MENTIONED = "mentioned"
    INTEGRATED = "integrated"
    CONFLICT = "conflict"
    DOMINANT = "dominant"


class PredictionChoice(Enum):
    """L1: 事件选择（4 个候选）"""
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class JustificationType(Enum):
    """L3: 决策理由类型（多标签）"""
    DUTY = "duty"
    TRUTH = "truth"
    RISK = "risk"
    TIME = "time"
    EMOTION = "emotion"
    RELATIONSHIP = "relationship"
    CURIOSITY = "curiosity"
    OTHER = "other"


class AnalysisSource(Enum):
    """SceneAnalysis 中 TR 的来源"""
    MEASURED = "measured"
    INFERRED = "inferred"
    DEFAULT = "default"


# ============================================================
# 值对象（Value Objects）
# ============================================================

@dataclass(frozen=True)
class CandidateScore:
    """Router 对某个策略的评分"""
    policy: PolicyType
    score: float
    blocked: bool = False
    reason: Optional[str] = None


@dataclass(frozen=True)
class PolicyConfig:
    """Policy 的执行配置"""
    prediction: PredictionMode
    realization: RealizationMode
    policy_type: PolicyType


@dataclass(frozen=True)
class ExecutionMetrics:
    """L0: Control Fidelity 的执行度量"""
    instruction_fidelity: float
    scene_compatibility: float
    execution_fidelity: float
    retry_count: int = 0
    retry_success: bool = True
    execution_time: float = 0.0
    raw_reason: Optional[str] = None


@dataclass(frozen=True)
class PropagationObservation:
    """四层传播观察结果（L1-L4）"""
    prediction: PredictionChoice
    prediction_confidence: float
    reasoning: ReasoningLevel
    reasoning_evidence: str
    justification: Tuple[JustificationType, ...]
    construction: str


@dataclass(frozen=True)
class Metadata:
    """元数据：时间戳、版本、环境信息"""
    artifact_version: str = "1.2"
    oracle_version: str = "v1.0"
    generated_at: datetime = field(default_factory=datetime.now)
    model_name: str = ""
    scene_id: Optional[str] = None


@dataclass(frozen=True)
class SceneAnalysis:
    """场景分析结果"""
    tr: float
    prediction_plasticity: float
    source: AnalysisSource
    confidence: float
    state_type: str = ""
    features: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class RouterDecision:
    """Router 的决策结果"""
    selected_policy: PolicyType
    candidate_scores: Tuple[CandidateScore, ...]
    policy_config: PolicyConfig
    confidence: float
    margin: float
    rationale: str
    raw_analysis: Optional[Dict[str, Any]] = None


# ============================================================
# Runtime Validation Artifact（SSOT）
# ============================================================

# LayerControlTargets 的导入（需要从 compiler 导入）
# 为避免循环导入，使用 forward reference 字符串
# 实际使用时，由 runtime/__init__.py 统一导出


@dataclass(frozen=True)
class RuntimeValidationArtifact:
    """
    Runtime 验证工件（唯一事实来源 / SSOT）
    
    只存储不可变事实，不存储派生指标。
    """
    # ---- 无默认值字段（必须提供） ----
    metadata: Metadata
    scene_analysis: SceneAnalysis
    router_decision: RouterDecision
    policy: PolicyConfig
    layer_targets: Any  # LayerControlTargets（编译后的 IR）
    execution: ExecutionMetrics
    propagation: PropagationObservation
    
    # ---- 有默认值字段（可选） ----
    prompt: Optional[str] = None
    raw_narrative: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        # layer_targets 的序列化
        if hasattr(self.layer_targets, 'to_dict'):
            layer_targets_dict = self.layer_targets.to_dict()
        else:
            layer_targets_dict = str(self.layer_targets)
        
        return {
            "metadata": {
                "artifact_version": self.metadata.artifact_version,
                "oracle_version": self.metadata.oracle_version,
                "generated_at": self.metadata.generated_at.isoformat(),
                "model_name": self.metadata.model_name,
                "scene_id": self.metadata.scene_id,
            },
            "scene_analysis": {
                "tr": self.scene_analysis.tr,
                "prediction_plasticity": self.scene_analysis.prediction_plasticity,
                "source": self.scene_analysis.source.value,
                "confidence": self.scene_analysis.confidence,
                "state_type": self.scene_analysis.state_type,
                "features": self.scene_analysis.features,
                "reason": self.scene_analysis.reason,
            },
            "router_decision": {
                "selected_policy": self.router_decision.selected_policy.value,
                "candidate_scores": [
                    {"policy": cs.policy.value, "score": cs.score, "blocked": cs.blocked, "reason": cs.reason}
                    for cs in self.router_decision.candidate_scores
                ],
                "policy_config": {
                    "prediction": self.policy.prediction.value,
                    "realization": self.policy.realization.value,
                    "policy_type": self.policy.policy_type.value,
                },
                "confidence": self.router_decision.confidence,
                "margin": self.router_decision.margin,
                "rationale": self.router_decision.rationale,
            },
            "layer_targets": layer_targets_dict,
            "prompt": self.prompt[:500] + "..." if self.prompt and len(self.prompt) > 500 else self.prompt,
            "execution": {
                "instruction_fidelity": self.execution.instruction_fidelity,
                "scene_compatibility": self.execution.scene_compatibility,
                "execution_fidelity": self.execution.execution_fidelity,
                "retry_count": self.execution.retry_count,
                "retry_success": self.execution.retry_success,
                "execution_time": self.execution.execution_time,
                "raw_reason": self.execution.raw_reason,
            },
            "propagation": {
                "prediction": self.propagation.prediction.value,
                "prediction_confidence": self.propagation.prediction_confidence,
                "reasoning": self.propagation.reasoning.value,
                "reasoning_evidence": self.propagation.reasoning_evidence,
                "justification": [j.value for j in self.propagation.justification],
                "construction": self.propagation.construction,
            },
        }


# ============================================================
# 导出
# ============================================================

__all__ = [
    # Enums
    "PolicyType",
    "PredictionMode",
    "RealizationMode",
    "ReasoningLevel",
    "PredictionChoice",
    "JustificationType",
    "AnalysisSource",
    # Value Objects
    "CandidateScore",
    "PolicyConfig",
    "ExecutionMetrics",
    "PropagationObservation",
    "Metadata",
    # Domain Models
    "SceneAnalysis",
    "RouterDecision",
    "RuntimeValidationArtifact",
]