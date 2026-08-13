# tests/regression/test_runtime_behavior.py

"""
Runtime 行为回归测试
使用 Golden Test 验证 Runtime 行为与预期一致
"""

import pytest
from packaging.version import Version

from src.capabilities import (
    CapabilityRegistry,
    CapabilitySpec,
    CapabilityMetadata,
    CapabilityRef,
)
from src.runtime.builder import RuntimeBuilder
from src.runtime.registry import SurfaceRegistry
from src.runtime.observation_compiler import ObservationCompiler
from src.surfaces.definition import (
    SurfaceDefinition,
    SurfaceMetadata,
    ObservationSpec,
    ValidationSpec,
    RepairSpec,
    PatternDefinition,
)


class KeywordCapability:
    def match(self, text: str, config: dict):
        results = []
        keywords = config.get("keywords", [])
        for kw in keywords:
            pos = text.find(kw)
            if pos != -1:
                results.append({
                    "pattern_type": "keyword",
                    "start": pos,
                    "end": pos + len(kw),
                    "text": kw,
                })
        return results


FIXED_DRAFT = """林逸的指节在袖中捏紧又松开三次。那封泛黄的密信在怀中发烫。
墨迹晕染处仍能辨认出"天机阁地底第七重"的批注。
风卷起几片零落桃花，落在那人玄色衣摆上。
"""

GOLDEN = [
    ("keyword", "密信"),
    ("keyword", "天机阁"),
]


def test_runtime_behavior_unchanged():
    """验证 Runtime 行为与 Golden 一致"""
    spec = CapabilitySpec(
        id="builtin.keyword",
        version=Version("1.0.0"),
        metadata=CapabilityMetadata(display_name="Keyword", description=""),
    )
    impl = KeywordCapability()

    registry = CapabilityRegistry(
        specs={"builtin.keyword": spec},
        implementations={"builtin.keyword": impl},
    )

    pattern = PatternDefinition(
        name="keyword_pattern",
        capability_ref=CapabilityRef.parse("builtin.keyword"),
        config={"keywords": ["密信", "天机阁"]},
    )

    surface = SurfaceDefinition(
        metadata=SurfaceMetadata(id="test", display_name="Test"),
        observation=ObservationSpec(patterns=(pattern,)),
        validation=ValidationSpec(layer_rules=()),
        repair=RepairSpec(repair_strategies=()),
    )

    catalog = SurfaceRegistry((surface,))
    builder = RuntimeBuilder(catalog, registry)
    snapshot = builder.build()

    compiler = ObservationCompiler()
    ir = compiler.compile(FIXED_DRAFT, snapshot)

    actual = [(p.pattern_type, p.text) for p in ir.patterns]
    assert actual == GOLDEN


def test_runtime_snapshot_has_capability_lookup():
    """验证 RuntimeSnapshot 持有 CapabilityLookup"""
    registry = CapabilityRegistry(specs={}, implementations={})
    catalog = SurfaceRegistry(())
    builder = RuntimeBuilder(catalog, registry)
    snapshot = builder.build()

    assert hasattr(snapshot, "capability_registry")
    from src.capabilities import CapabilityLookup
    assert isinstance(snapshot.capability_registry, CapabilityLookup)
