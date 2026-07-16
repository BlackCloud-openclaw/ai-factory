"""
SurfaceDefinition - 纯声明式配置
Surface 描述 Capability，不描述 Composition
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from types import MappingProxyType


# ============================================================
# 1. 基础数据结构：Pattern / Metric / Rule / Strategy
# ============================================================

@dataclass(frozen=True)
class PatternDefinition:
    """声明一个 Pattern，由 ObservationCompiler 解释"""
    name: str
    matcher: str  # "keyword" | "regex" | "quotation"
    config: Dict[str, Any] = field(default_factory=dict)


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
    """
    声明一个修复策略，由 EditCompiler 解释
    使用结构化 trigger，而非字符串 DSL
    """
    target_layer: str
    trigger: str  # "non_compliant" | "ratio_low" | "turns_low" | "missing_pattern"
    operation: str  # "REPLACE_SENTENCE" | "INSERT_AFTER" | "INSERT_BEFORE"
    payload_type: str  # "combined" | "dialogue_marker" | "state_keyword" | "logic_marker"


# ============================================================
# 2. Spec 分组（每个 Surface 的组成部分）
# ============================================================

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


# ============================================================
# 3. SurfaceMetadata（仅用于标识，不参与行为）
# ============================================================

@dataclass(frozen=True)
class SurfaceMetadata:
    """Surface 元数据，仅用于标识和展示"""
    id: str
    display_name: str


# ============================================================
# 4. SurfaceDefinition（纯声明，不含 Composition 信息）
# ============================================================

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