"""
Echo Surface - 测试夹具（Fixture）
用于 Phase 7A-2 聚合测试，不作为产品 Surface 使用
"""

from src.surfaces.definition import (
    SurfaceDefinition,
    SurfaceMetadata,
    ObservationSpec,
    ValidationSpec,
    RepairSpec,
    PatternDefinition,
    LayerRule,
    RepairStrategy,
)


ECHO_PATTERNS = (
    PatternDefinition(
        name="echo_marker",
        matcher="keyword",
        config={"keywords": ["echo", "Echo", "ECHO"]},
    ),
)

ECHO_LAYER_RULES = (
    LayerRule(
        layer="echo",
        required_types=["echo_marker"],
    ),
)

ECHO_REPAIR_STRATEGIES = (
    RepairStrategy(
        target_layer="echo",
        trigger="non_compliant",
        operation="INSERT_AFTER",
        payload_type="echo_marker",
    ),
)

EchoSurface = SurfaceDefinition(
    metadata=SurfaceMetadata(
        id="echo",
        display_name="Echo Surface",
    ),
    observation=ObservationSpec(
        patterns=ECHO_PATTERNS,
    ),
    validation=ValidationSpec(
        layer_rules=ECHO_LAYER_RULES,
    ),
    repair=RepairSpec(
        repair_strategies=ECHO_REPAIR_STRATEGIES,
    ),
)