# tests/unit/capabilities/runtime/test_registry.py

import pytest
from packaging.version import Version

from src.capabilities import CapabilitySpec, CapabilityMetadata
from src.capabilities.runtime import (
    RuntimeCapabilityRegistry,
    RuntimeCapability,
)


class MockRuntimeCapability:
    def __init__(self, service):
        self._service = service

    def get(self):
        return self._service


class TestRuntimeCapabilityRegistry:

    def setup_method(self):
        self.spec = CapabilitySpec(
            id="test.runtime.service",
            version=Version("1.0.0"),
            metadata=CapabilityMetadata(
                display_name="Test Service",
                description="Test runtime service",
                tags=("test",),
            ),
        )
        self.capability = MockRuntimeCapability("test_service")
        self.registry = RuntimeCapabilityRegistry()

    def test_register_and_require(self):
        self.registry.register(self.spec, self.capability)
        retrieved = self.registry.require("test.runtime.service")
        assert retrieved is self.capability
        assert retrieved.get() == "test_service"

    def test_require_unknown_raises(self):
        with pytest.raises(KeyError, match="Runtime capability not found"):
            self.registry.require("unknown.service")

    def test_register_duplicate_raises(self):
        self.registry.register(self.spec, self.capability)
        with pytest.raises(ValueError, match="already registered"):
            self.registry.register(self.spec, self.capability)

    def test_has_returns_correct(self):
        assert self.registry.has("test.runtime.service") is False
        self.registry.register(self.spec, self.capability)
        assert self.registry.has("test.runtime.service") is True

    def test_list_ids(self):
        self.registry.register(self.spec, self.capability)
        ids = self.registry.list_ids()
        assert "test.runtime.service" in ids

    def test_freeze_prevents_registration(self):
        self.registry.register(self.spec, self.capability)
        frozen = self.registry.freeze()

        with pytest.raises(RuntimeError, match="RuntimeCapabilityRegistry is frozen"):
            self.registry.register(self.spec, self.capability)

        retrieved = frozen.require("test.runtime.service")
        assert retrieved is self.capability

    def test_freeze_empty_raises(self):
        empty_registry = RuntimeCapabilityRegistry()
        with pytest.raises(RuntimeError, match="Cannot freeze empty"):
            empty_registry.freeze()

    def test_frozen_has_and_list_ids(self):
        self.registry.register(self.spec, self.capability)
        frozen = self.registry.freeze()

        assert frozen.has("test.runtime.service") is True
        assert frozen.has("unknown.service") is False
        assert "test.runtime.service" in frozen.list_ids()

    def test_multiple_registrations(self):
        spec2 = CapabilitySpec(
            id="test.runtime.service2",  # 修正：capability_id → id
            version=Version("1.0.0"),
            metadata=CapabilityMetadata(
                display_name="Test Service 2",
                description="Another test service",
                tags=("test",),
            ),
        )
        cap2 = MockRuntimeCapability("service2")

        self.registry.register(self.spec, self.capability)
        self.registry.register(spec2, cap2)

        assert self.registry.require("test.runtime.service").get() == "test_service"
        assert self.registry.require("test.runtime.service2").get() == "service2"

        frozen = self.registry.freeze()
        assert len(frozen.list_ids()) == 2