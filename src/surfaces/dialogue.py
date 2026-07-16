"""
Dialogue Surface - Phase 7B 第一个验证插件
验证 Runtime 能够通过新增 Surface 扩展能力，而无需修改核心代码

第一轮：仅验证框架机制
- Observation: dialogue_marker（引号检测）
- Validation: dialogue_exists（至少有一段对话）
- Repair: INSERT_DIALOGUE
"""

from src.surfaces.definition import (
    SurfaceDefinition,
    SurfaceMetadata,
    ObservationSpec,
    ValidationSpec,
    RepairSpec,
    PatternDefinition,
    LayerRule,
    MetricDefinition,
    RepairStrategy,
)
from src.capabilities.ids import Matchers, Metrics, Repairs, Triggers


# ============================================================
# 1. Observation Patterns
# ============================================================

DIALOGUE_PATTERNS = (
    PatternDefinition(
        name="dialogue_marker",
        matcher=Matchers.QUOTATION,
        config={},  # 留空，使用默认正则
    ),
)

# ============================================================
# 2. Validation Layer Rules
# ============================================================

DIALOGUE_LAYER_RULES = (
    LayerRule(
        layer="dialogue",
        required_types=["dialogue_marker"],
        metrics=(
            MetricDefinition(
                name="dialogue_exists",
                operator="gte",
                target=1,
            ),
        )
    ),
)


# ============================================================
# 3. Repair Strategies
# ============================================================

DIALOGUE_REPAIR_STRATEGIES = (
    RepairStrategy(
        target_layer="dialogue",
        trigger="non_compliant",  # 当 dialogue 层不合规时触发
        operation=Repairs.INSERT_DIALOGUE,
        payload_type="dialogue_marker",
    ),
)


# ============================================================
# 4. SurfaceDefinition
# ============================================================

DialogueSurface = SurfaceDefinition(
    metadata=SurfaceMetadata(
        id="dialogue",
        display_name="对话控制",
    ),
    observation=ObservationSpec(
        patterns=DIALOGUE_PATTERNS,
    ),
    validation=ValidationSpec(
        layer_rules=DIALOGUE_LAYER_RULES,
    ),
    repair=RepairSpec(
        repair_strategies=DIALOGUE_REPAIR_STRATEGIES,
    ),
)