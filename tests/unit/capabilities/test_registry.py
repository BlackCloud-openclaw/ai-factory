# tests/unit/capabilities/test_registry.py

import pytest
from packaging.version import Version

from src.capabilities import (
    CapabilitySpec,
    CapabilityMetadata,
    CapabilityRef,
    CapabilityRegistry,
    CapabilityImplementation,
    CapabilityNotFoundError,
    CapabilityVersionError,
    CapabilityImplementationError,
)


class FakeImplementation:
    def match(self, text: str, config: dict):
        return [{"type": "fake"}]


class NotAProtocolImplementation:
    pass


def test_version_comparison():
    v1 = Version("1.10.0")
    v2 = Version("1.2.0")
    assert v1 > v2
    assert v2 < v1
    assert Version("1.0.0") == Version("1.0.0")


def test_capability_implementation_protocol():
    assert isinstance(FakeImplementation(), CapabilityImplementation)
    assert not isinstance(NotAProtocolImplementation(), CapabilityImplementation)


def test_capability_ref_parse():
    ref = CapabilityRef.parse("builtin.keyword@1.0.0")
    assert ref.id == "builtin.keyword"
    assert ref.version == Version("1.0.0")

    ref2 = CapabilityRef.parse("builtin.keyword")
    assert ref2.id == "builtin.keyword"
    assert ref2.version is None

    with pytest.raises(ValueError):
        CapabilityRef.parse("")
    with pytest.raises(ValueError):
        CapabilityRef.parse("@1.0.0")
    with pytest.raises(ValueError):
        CapabilityRef.parse("builtin.keyword@")


def test_capability_ref_str():
    ref = CapabilityRef("builtin.keyword", Version("1.0.0"))
    assert str(ref) == "builtin.keyword@1.0.0"
    ref2 = CapabilityRef("builtin.keyword")
    assert str(ref2) == "builtin.keyword"


def test_registry_get_spec():
    spec = CapabilitySpec(
        id="test.cap",
        version=Version("1.0.0"),
        metadata=CapabilityMetadata(display_name="Test", description=""),
    )
    registry = CapabilityRegistry(
        specs={"test.cap": spec},
        implementations={"test.cap": FakeImplementation()},
    )
    result = registry.get_spec(CapabilityRef("test.cap"))
    assert result.id == "test.cap"


def test_registry_get_spec_with_version():
    spec = CapabilitySpec(
        id="test.cap",
        version=Version("1.0.0"),
        metadata=CapabilityMetadata(display_name="Test", description=""),
    )
    registry = CapabilityRegistry(
        specs={"test.cap": spec},
        implementations={"test.cap": FakeImplementation()},
    )
    result = registry.get_spec(CapabilityRef("test.cap", version=Version("1.0.0")))
    assert result.id == "test.cap"


def test_registry_get_spec_not_found():
    registry = CapabilityRegistry(specs={}, implementations={})
    with pytest.raises(CapabilityNotFoundError):
        registry.get_spec(CapabilityRef("nonexistent"))


def test_registry_get_spec_version_mismatch():
    spec = CapabilitySpec(
        id="test.cap",
        version=Version("1.0.0"),
        metadata=CapabilityMetadata(display_name="Test", description=""),
    )
    registry = CapabilityRegistry(
        specs={"test.cap": spec},
        implementations={"test.cap": FakeImplementation()},
    )
    with pytest.raises(CapabilityVersionError):
        registry.get_spec(CapabilityRef("test.cap", version=Version("2.0.0")))


def test_registry_get_impl():
    spec = CapabilitySpec(
        id="test.cap",
        version=Version("1.0.0"),
        metadata=CapabilityMetadata(display_name="Test", description=""),
    )
    impl = FakeImplementation()
    registry = CapabilityRegistry(
        specs={"test.cap": spec},
        implementations={"test.cap": impl},
    )
    result = registry.get_impl(CapabilityRef("test.cap"))
    assert result is impl


def test_registry_get_impl_not_protocol():
    spec = CapabilitySpec(
        id="test.cap",
        version=Version("1.0.0"),
        metadata=CapabilityMetadata(display_name="Test", description=""),
    )
    non_protocol = NotAProtocolImplementation()
    registry = CapabilityRegistry(
        specs={"test.cap": spec},
        implementations={"test.cap": non_protocol},
    )
    with pytest.raises(CapabilityImplementationError, match="does not implement"):
        registry.get_impl(CapabilityRef("test.cap"))


def test_registry_init_missing_impl():
    spec = CapabilitySpec(
        id="test.cap",
        version=Version("1.0.0"),
        metadata=CapabilityMetadata(display_name="Test", description=""),
    )
    with pytest.raises(CapabilityImplementationError, match="has Spec but no Implementation"):
        CapabilityRegistry(specs={"test.cap": spec}, implementations={})


def test_registry_immutable():
    spec = CapabilitySpec(
        id="test.cap",
        version=Version("1.0.0"),
        metadata=CapabilityMetadata(display_name="Test", description=""),
    )
    registry = CapabilityRegistry(
        specs={"test.cap": spec},
        implementations={"test.cap": FakeImplementation()},
    )
    with pytest.raises(TypeError):
        registry._specs["new"] = spec  # type: ignore


def test_registry_has():
    spec = CapabilitySpec(
        id="test.cap",
        version=Version("1.0.0"),
        metadata=CapabilityMetadata(display_name="Test", description=""),
    )
    registry = CapabilityRegistry(
        specs={"test.cap": spec},
        implementations={"test.cap": FakeImplementation()},
    )
    assert registry.has(CapabilityRef("test.cap")) is True
    assert registry.has(CapabilityRef("nonexistent")) is False
    assert registry.has(CapabilityRef("test.cap", version=Version("2.0.0"))) is True


def test_registry_find_spec():
    spec = CapabilitySpec(
        id="test.cap",
        version=Version("1.0.0"),
        metadata=CapabilityMetadata(display_name="Test", description=""),
    )
    registry = CapabilityRegistry(
        specs={"test.cap": spec},
        implementations={"test.cap": FakeImplementation()},
    )
    assert registry.find_spec(CapabilityRef("test.cap")) is not None
    assert registry.find_spec(CapabilityRef("nonexistent")) is None
    assert registry.find_spec(CapabilityRef("test.cap", version=Version("2.0.0"))) is None


def test_registry_list_specs_returns_tuple():
    spec = CapabilitySpec(
        id="test.cap",
        version=Version("1.0.0"),
        metadata=CapabilityMetadata(display_name="Test", description=""),
    )
    registry = CapabilityRegistry(
        specs={"test.cap": spec},
        implementations={"test.cap": FakeImplementation()},
    )
    result = registry.list_specs()
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0].id == "test.cap"


def test_registry_list_impls_returns_tuple():
    spec = CapabilitySpec(
        id="test.cap",
        version=Version("1.0.0"),
        metadata=CapabilityMetadata(display_name="Test", description=""),
    )
    impl = FakeImplementation()
    registry = CapabilityRegistry(
        specs={"test.cap": spec},
        implementations={"test.cap": impl},
    )
    result = registry.list_impls()
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0] is impl
