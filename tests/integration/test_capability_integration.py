# tests/integration/test_capability_integration.py

import pytest
from packaging.version import Version

from src.capabilities import (
    CapabilitySpec,
    CapabilityMetadata,
    CapabilityRef,
    CapabilityRegistry,
    CapabilityImplementation,
    CapabilityLookup,
    CapabilityExecutionError,
)
from src.runtime.builder import RuntimeBuilder
from src.runtime.registry import SurfaceRegistry
from src.runtime.observation_compiler import ObservationCompiler
from src.runtime.snapshot import RuntimeConfig
from src.surfaces.definition import (
    SurfaceDefinition,
    SurfaceMetadata,
    ObservationSpec,
    ValidationSpec,
    RepairSpec,
    PatternDefinition,
    LayerRule,
)


class FakeCapability:
    def match(self, text: str, config: dict):
        return [{"type": "fake", "start": 0, "end": 4, "text": text[:4]}]


class BrokenCapability:
    def match(self, text: str, config: dict):
        raise CapabilityExecutionError("simulated failure")


def test_registry_implements_protocol():
    registry = CapabilityRegistry(specs={}, implementations={})
    assert isinstance(registry, CapabilityLookup)


def test_has_version_semantics():
    spec = CapabilitySpec(
        id="test.cap",
        version=Version("1.0.0"),
        metadata=CapabilityMetadata(display_name="Test", description=""),
    )
    registry = CapabilityRegistry(
        specs={"test.cap": spec},
        implementations={"test.cap": FakeCapability()},
    )

    assert registry.has(CapabilityRef("test.cap")) is True
    assert registry.has(CapabilityRef("test.cap", version=Version("2.0.0"))) is True
    assert registry.has(CapabilityRef("nonexistent")) is False


def test_runtime_uses_capability_lookup():
    spec = CapabilitySpec(
        id="test.fake",
        version=Version("1.0.0"),
        metadata=CapabilityMetadata(
            display_name="Fake",
            description="Fake capability for testing",
        ),
    )
    impl = FakeCapability()

    registry = CapabilityRegistry(
        specs={"test.fake": spec},
        implementations={"test.fake": impl},
    )

    pattern = PatternDefinition(
        name="fake_pattern",
        capability_ref=CapabilityRef.parse("test.fake"),
        config={},
    )

    surface = SurfaceDefinition(
        metadata=SurfaceMetadata(id="test_surface", display_name="Test"),
        observation=ObservationSpec(patterns=(pattern,)),
        validation=ValidationSpec(layer_rules=()),
        repair=RepairSpec(repair_strategies=()),
    )

    catalog = SurfaceRegistry((surface,))
    builder = RuntimeBuilder(catalog, registry)
    snapshot = builder.build()

    compiler = ObservationCompiler()
    ir = compiler.compile("test draft", snapshot)

    assert len(ir.patterns) == 1
    assert ir.patterns[0].pattern_type == "fake"
    assert ir.patterns[0].text == "test"


def test_missing_capability_skips_pattern():
    registry = CapabilityRegistry(specs={}, implementations={})

    pattern = PatternDefinition(
        name="missing_pattern",
        capability_ref=CapabilityRef.parse("nonexistent"),
        config={},
    )

    surface = SurfaceDefinition(
        metadata=SurfaceMetadata(id="test_surface", display_name="Test"),
        observation=ObservationSpec(patterns=(pattern,)),
        validation=ValidationSpec(layer_rules=()),
        repair=RepairSpec(repair_strategies=()),
    )

    catalog = SurfaceRegistry((surface,))
    builder = RuntimeBuilder(catalog, registry)
    snapshot = builder.build()

    compiler = ObservationCompiler()
    ir = compiler.compile("test draft", snapshot)

    assert len(ir.patterns) == 0


def test_broken_capability_does_not_crash_runtime():
    spec = CapabilitySpec(
        id="test.broken",
        version=Version("1.0.0"),
        metadata=CapabilityMetadata(display_name="Broken", description=""),
    )
    broken_impl = BrokenCapability()

    normal_spec = CapabilitySpec(
        id="test.normal",
        version=Version("1.0.0"),
        metadata=CapabilityMetadata(display_name="Normal", description=""),
    )
    normal_impl = FakeCapability()  # 注意：这里返回 "fake"

    registry = CapabilityRegistry(
        specs={
            "test.broken": spec,
            "test.normal": normal_spec,
        },
        implementations={
            "test.broken": broken_impl,
            "test.normal": normal_impl,
        },
    )

    broken_pattern = PatternDefinition(
        name="broken_pattern",
        capability_ref=CapabilityRef.parse("test.broken"),
        config={},
    )
    normal_pattern = PatternDefinition(
        name="normal_pattern",
        capability_ref=CapabilityRef.parse("test.normal"),
        config={},
    )

    surface = SurfaceDefinition(
        metadata=SurfaceMetadata(id="test_surface", display_name="Test"),
        observation=ObservationSpec(patterns=(broken_pattern, normal_pattern)),
        validation=ValidationSpec(layer_rules=()),
        repair=RepairSpec(repair_strategies=()),
    )

    catalog = SurfaceRegistry((surface,))
    builder = RuntimeBuilder(catalog, registry)
    snapshot = builder.build()

    compiler = ObservationCompiler()
    ir = compiler.compile("test draft", snapshot)

    # Broken Pattern 被跳过，Normal Pattern 仍被提取
    assert len(ir.patterns) == 1
    # 注意：normal_impl 是 FakeCapability，返回 "fake"
    assert ir.patterns[0].pattern_type == "fake"


def test_observation_compiler_has_no_matcher_registry():
    compiler = ObservationCompiler()
    assert not hasattr(compiler, "_matcher_registry"), \
        "ObservationCompiler should not have _matcher_registry"