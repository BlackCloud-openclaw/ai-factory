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
    """
    传播策略类型（Propagation Policy）
    
    CONSERVATIVE: 保守策略（DISABLED + ENHANCED）
    ADAPTIVE:    自适应策略（ASSIST + ENHANCED）
    AGGRESSIVE:  激进策略（PRIMARY + ENHANCED）
    """
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"
    AGGRESSIVE = "aggressive"


class PredictionMode(Enum):
    """
    Prediction Layer 的控制模式
    
    DISABLED:  不尝试改变事件选择
    ASSIST:    State 作为辅助参考
    PRIMARY:   State 主导事件选择
    """
    DISABLED = "disabled"
    ASSIST = "assist"
    PRIMARY = "primary"


class RealizationMode(Enum):
    """
    Realization Layer 的控制模式
    
    NONE:      不注入 State
    NORMAL:    正常注入 State
    ENHANCED:  增强注入 State（State 在叙事中占据重要位置）
    """
    NONE = "none"
    NORMAL = "normal"
    ENHANCED = "enhanced"


class ReasoningLevel(Enum):
    """
    L2: State Integration 层级
    
    IGNORED:    State 完全没有被使用
    MENTIONED:  提到 State，但决策逻辑不受影响
    INTEGRATED: State 成为决策依据之一
    CONFLICT:   State 进入推理，但被更强理由压制
    DOMINANT:   State 主导整个决策
    """
    IGNORED = "ignored"
    MENTIONED = "mentioned"
    INTEGRATED = "integrated"
    CONFLICT = "conflict"
    DOMINANT = "dominant"


class PredictionChoice(Enum):
    """
    L1: 事件选择（4 个候选）
    
    A: 立即返回宗门，遵守师命
    B: 立即前往禁地，探查灵力波动
    C: 立即在原地留下标记，等待同伴
    D: 立即登上高处，观察禁地方向再决定
    """
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class JustificationType(Enum):
    """
    L3: 决策理由类型（多标签）
    
    DUTY:        责任、师命、承诺、契约
    TRUTH:       好奇心、真相、探索、直觉
    RISK:        风险评估、避险、谨慎
    TIME:        时间紧迫性、截止日期
    EMOTION:     情感驱动
    RELATIONSHIP:人际关系驱动
    CURIOSITY:   好奇心（与 TRUTH 区分，更偏向好奇而非真相）
    OTHER:       其他
    """
    DUTY = "duty"
    TRUTH = "truth"
    RISK = "risk"
    TIME = "time"
    EMOTION = "emotion"
    RELATIONSHIP = "relationship"
    CURIOSITY = "curiosity"
    OTHER = "other"


class AnalysisSource(Enum):
    """
    SceneAnalysis 中 TR 的来源
    
    MEASURED:  实验测量值（高置信度）
    INFERRED:  特征推断值（中等置信度）
    DEFAULT:   默认值（低置信度）
    """
    MEASURED = "measured"
    INFERRED = "inferred"
    DEFAULT = "default"


# ============================================================
# 值对象（Value Objects）
# ============================================================

@dataclass(frozen=True)
class CandidateScore:
    """
    Router 对某个策略的评分
    
    Attributes:
        policy: 策略类型
        score: 兼容性分数（0-1），越高表示策略越适合当前场景
        blocked: 该策略是否被明确排除（如因为与 TR 不兼容）
        reason: 可选的理由说明
    """
    policy: PolicyType
    score: float
    blocked: bool = False
    reason: Optional[str] = None


@dataclass(frozen=True)
class PolicyConfig:
    """
    Policy 的执行配置
    
    由 Router 输出，Writer 执行。
    
    Attributes:
        prediction: Prediction Layer 的控制模式
        realization: Realization Layer 的控制模式
        policy_type: 策略类型（便于追踪和统计）
    """
    prediction: PredictionMode
    realization: RealizationMode
    policy_type: PolicyType


@dataclass(frozen=True)
class ExecutionMetrics:
    """
    L0: Control Fidelity 的执行度量
    
    Attributes:
        instruction_fidelity: Writer 是否按 Policy 指令执行（0-1）
        scene_compatibility: Policy 是否与场景的 TR 兼容（0-1）
        execution_fidelity: 综合执行保真度 = instruction_fidelity × scene_compatibility
        retry_count: 重试次数
        retry_success: 重试是否成功
        execution_time: 执行耗时（秒）
        raw_reason: 诊断理由（可选）
    """
    instruction_fidelity: float
    scene_compatibility: float
    execution_fidelity: float
    retry_count: int = 0
    retry_success: bool = True
    execution_time: float = 0.0
    raw_reason: Optional[str] = None


@dataclass(frozen=True)
class PropagationObservation:
    """
    四层传播观察结果（L1-L4）
    
    Attributes:
        prediction: L1 事件选择
        prediction_confidence: L1 置信度（1.0/0.7/0.5）
        reasoning: L2 State Integration 层级
        reasoning_evidence: L2 证据句
        justification: L3 决策理由（多标签）
        construction: L4 叙事构建描述
    """
    prediction: PredictionChoice
    prediction_confidence: float
    reasoning: ReasoningLevel
    reasoning_evidence: str
    justification: Tuple[JustificationType, ...]
    construction: str


@dataclass(frozen=True)
class Metadata:
    """
    元数据：时间戳、版本、环境信息
    
    Attributes:
        artifact_version: Artifact 版本号
        oracle_version: Oracle 版本号
        generated_at: 生成时间
        model_name: 使用的 LLM 模型名称
        scene_id: 场景标识（可选）
    """
    artifact_version: str = "1.0"
    oracle_version: str = "v1.0"
    generated_at: datetime = field(default_factory=datetime.now)
    model_name: str = ""
    scene_id: Optional[str] = None


# ============================================================
# 领域对象（Domain Models）
# ============================================================

@dataclass(frozen=True)
class SceneAnalysis:
    """
    场景分析结果
    
    Scene Analyzer 的输出，Router 的输入。
    包含场景的 TR、Prediction Plasticity 等特征。
    
    Attributes:
        tr: Transition Rigidity（0-1）
        prediction_plasticity: Prediction Layer 的可塑性（0-1），
                              由 TR 派生，描述事件选择的可变空间
        source: TR 的来源
        confidence: TR 的置信度（0-1）
        state_type: 注入的 State 类型（如 "exploration", "task"）
        features: 原始场景特征（用于调试）
        reason: 分析理由
    """
    tr: float
    prediction_plasticity: float
    source: AnalysisSource
    confidence: float
    state_type: str = ""
    features: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class RouterDecision:
    """
    Router 的决策结果
    
    Router 的完整输出，包含决策轨迹和可解释性信息。
    
    Attributes:
        selected_policy: 选中的策略类型
        candidate_scores: 所有候选策略的评分
        policy_config: 策略的执行配置（展开后的 Policy）
        confidence: 对选中策略的置信度（0-1），即最高分
        margin: 最高分与次高分的差值（0-1），反映决策确定性
        rationale: 人类可读的决策理由
        raw_analysis: 原始分析数据（可选，用于调试）
    """
    selected_policy: PolicyType
    candidate_scores: Tuple[CandidateScore, ...]
    policy_config: PolicyConfig
    confidence: float
    margin: float
    rationale: str
    raw_analysis: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class RuntimeValidationArtifact:
    """
    Runtime 验证工件（唯一事实来源 / SSOT）
    
    这是 Runtime 系统的统一输出，所有指标均由 Artifact 派生计算。
    
    设计原则：
    - 只存储不可变事实（Fact），不存储派生指标（Score/Gap）
    - 所有 dataclass 均为 frozen=True
    - 可序列化为 JSON/YAML 用于持久化和调试
    
    Attributes:
        metadata: 元数据（版本、时间、模型等）
        scene_analysis: Scene Analyzer 的分析结果
        router_decision: Router 的决策结果
        policy: Policy 执行配置（从 RouterDecision 提取，便于直接访问）
        execution: Execution Metrics（L0）
        propagation: Propagation Observation（L1-L4）
        raw_narrative: 原始生成的叙事文本（可选，用于调试）
    """
    metadata: Metadata
    scene_analysis: SceneAnalysis
    router_decision: RouterDecision
    policy: PolicyConfig
    execution: ExecutionMetrics
    propagation: PropagationObservation
    raw_narrative: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典，便于 JSON/YAML 输出"""
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
# 便捷构造函数
# ============================================================

def create_candidate_score(
    policy: PolicyType,
    score: float,
    blocked: bool = False,
    reason: Optional[str] = None,
) -> CandidateScore:
    """便捷构造函数"""
    return CandidateScore(policy=policy, score=score, blocked=blocked, reason=reason)


def create_policy_config(
    prediction: PredictionMode,
    realization: RealizationMode,
    policy_type: PolicyType,
) -> PolicyConfig:
    """便捷构造函数"""
    return PolicyConfig(prediction=prediction, realization=realization, policy_type=policy_type)


def create_propagation_observation(
    prediction: PredictionChoice,
    prediction_confidence: float,
    reasoning: ReasoningLevel,
    reasoning_evidence: str,
    justification: List[JustificationType],
    construction: str,
) -> PropagationObservation:
    """便捷构造函数"""
    return PropagationObservation(
        prediction=prediction,
        prediction_confidence=prediction_confidence,
        reasoning=reasoning,
        reasoning_evidence=reasoning_evidence,
        justification=tuple(justification),
        construction=construction,
    )


# ============================================================
# 导出（Export）
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
    # Constructors
    "create_candidate_score",
    "create_policy_config",
    "create_propagation_observation",
]