# src/surfaces/reasoning.py

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
from src.capabilities import CapabilityRef


# ============================================================
# 1. Observation Patterns
# ============================================================

STATE_KEYWORDS = [
    # 原有
    "密信", "血契", "玉牌", "玉佩", "烙印", "印记", "罗盘", "青铜",
    "药水", "符纸", "龙纹", "胎记", "黑曜石", "铜铃",
    # 扩展
    "法器", "符箓", "阵眼", "灵脉", "封印", "禁制", "阵法",
    "灵田", "灵气", "灵根", "功法", "残卷", "秘笈", "密卷",
    "令牌", "剑穗", "香灰", "灰烬", "血玉", "玄铁",
    "藏经阁", "贡献簿", "断碑", "机关",
]

LOGIC_KEYWORDS = [
    "忽然想起", "忽然意识到", "这意味着", "是因为", "所以", "因此",
    "这意味着", "突然察觉", "意识到", "发现", "推断出",
    "为了", "以免", "如果", "则", "否则", "既然", "因为", "于是"
]

REASONING_PATTERNS = (
    PatternDefinition(
        name="state_keyword",
        capability_ref=CapabilityRef.parse("builtin.keyword"),
        config={"keywords": STATE_KEYWORDS}
    ),
    PatternDefinition(
        name="logic_marker",
        capability_ref=CapabilityRef.parse("builtin.keyword"),
        config={"keywords": LOGIC_KEYWORDS}
    ),
)


# ============================================================
# 2. Validation Layer Rules
# ============================================================

REASONING_LAYER_RULES = (
    LayerRule(
        layer="reasoning",
        required_types=["logic_marker", "state_keyword"],
    ),
    LayerRule(
        layer="justification",
        required_types=["logic_marker", "state_keyword"],
    ),
    LayerRule(
        layer="construction",
        required_types=["state_keyword"],
    ),
    LayerRule(
        layer="prediction",
        required_types=["state_keyword"],
    ),
)


# ============================================================
# 3. Repair Strategies
# ============================================================

REASONING_REPAIR_STRATEGIES = (
    RepairStrategy(
        target_layer="reasoning",
        trigger="non_compliant",
        operation="REPLACE_SENTENCE",
        payload_type="combined",
    ),
    RepairStrategy(
        target_layer="justification",
        trigger="non_compliant",
        operation="REPLACE_SENTENCE",
        payload_type="combined",
    ),
)


# ============================================================
# 4. SurfaceDefinition
# ============================================================

ReasoningSurface = SurfaceDefinition(
    metadata=SurfaceMetadata(
        id="reasoning",
        display_name="推理与论证",
    ),
    observation=ObservationSpec(
        patterns=REASONING_PATTERNS,
    ),
    validation=ValidationSpec(
        layer_rules=REASONING_LAYER_RULES,
    ),
    repair=RepairSpec(
        repair_strategies=REASONING_REPAIR_STRATEGIES,
    ),
)