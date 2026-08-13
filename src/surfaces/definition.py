# src/surfaces/definition.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from src.capabilities import CapabilityRef


@dataclass(frozen=True)
class SurfaceMetadata:
    """Surface 元数据，仅用于标识和展示"""
    id: str
    display_name: str


@dataclass(frozen=True)
class PatternDefinition:
    """声明一个 Pattern，由 ObservationCompiler 解释"""
    name: str
    # 旧字段（已废弃，由 Loader 迁移到 capability_ref）
    matcher: Optional[str] = field(
        default=None,
        metadata={"deprecated": True},
    )
    config: Dict[str, Any] = field(default_factory=dict)
    # 新字段（推荐使用）
    capability_ref: Optional[CapabilityRef] = None


@dataclass(frozen=True)
class MetricDefinition:
    """声明一个度量指标，由 Validator 解释"""
    name: str
    operator: str  # "between" | "gt" | "lt" | "gte" | "lte" | "eq"
    min: Optional[float] = None
    max: Optional[float] = None
    target: Optional[float] = None


@dataclass(frozen=True)
class LayerRule:
    """声明一个 Layer 规则，由 Validator 解释"""
    layer: str
    required_types: List[str] = field(default_factory=list)
    metrics: List[MetricDefinition] = field(default_factory=list)


@dataclass(frozen=True)
class RepairStrategy:
    """声明一个修复策略，由 EditCompiler 解释"""
    target_layer: str
    trigger: str  # "non_compliant" | "ratio_low" | "turns_low" | "missing_pattern"
    operation: str  # "REPLACE_SENTENCE" | "INSERT_AFTER" | "INSERT_BEFORE"
    payload_type: str  # "combined" | "dialogue_marker" | "state_keyword" | "logic_marker"


@dataclass(frozen=True)
class ObservationSpec:
    """Observation 能力：从文本中提取哪些 Pattern"""
    patterns: Tuple[PatternDefinition, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ValidationSpec:
    """Validation 能力：如何判定合规"""
    layer_rules: Tuple[LayerRule, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RepairSpec:
    """Repair 能力：如何修复不合规"""
    repair_strategies: Tuple[RepairStrategy, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SurfaceDefinition:
    """
    Surface 的完整定义 - 纯声明式配置
    
    关键约束：
    - 不包含执行顺序、优先级、依赖关系（由 Builder 负责）
    - 不包含回调或 Python 行为（Compiler 解释声明）
    - 不可变（frozen dataclass）
    """
    metadata: SurfaceMetadata
    observation: ObservationSpec
    validation: ValidationSpec
    repair: RepairSpec