# src/surfaces/dialogue.py

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
from src.capabilities import CapabilityRef


# ============================================================
# 1. Observation Patterns
# ============================================================

DIALOGUE_PATTERNS = (
    PatternDefinition(
        name="dialogue_marker",
        capability_ref=CapabilityRef.parse("builtin.quotation"),
        config={},
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
        trigger="non_compliant",
        operation="INSERT_DIALOGUE",
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