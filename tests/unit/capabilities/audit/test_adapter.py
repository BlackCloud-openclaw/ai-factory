# tests/unit/capabilities/audit/test_adapter.py

import pytest
from src.capabilities.audit import (
    AuditCapabilityAdapter,
    AuditCapability,
    AUDIT_COORDINATOR_SPEC,
)
from src.writing.audit import AuditConfig, MemoryPayloadResolver


class TestAuditCapabilityAdapter:

    def test_adapter_creates_capability(self):
        cap = AuditCapabilityAdapter.create()
        assert isinstance(cap, AuditCapability)

        coordinator = cap.get()
        assert coordinator is not None
        assert hasattr(coordinator, "start")
        assert hasattr(coordinator, "audit")

    def test_capability_returns_coordinator(self):
        cap = AuditCapabilityAdapter.create()
        c1 = cap.get()
        c2 = cap.get()
        assert c1 is c2  # 同一个实例

    def test_adapter_with_custom_resolver(self):
        resolver = MemoryPayloadResolver()
        cap = AuditCapabilityAdapter.create(resolver=resolver)
        coordinator = cap.get()
        assert coordinator._resolver is resolver

    def test_adapter_with_custom_config(self):
        config = AuditConfig(enabled=False)
        cap = AuditCapabilityAdapter.create(config=config)
        coordinator = cap.get()
        assert coordinator._config.enabled is False

    def test_spec_consistent(self):
        assert AUDIT_COORDINATOR_SPEC.capability_id == "builtin.audit.coordinator"
        assert AUDIT_COORDINATOR_SPEC.version == "1.0"
        assert AUDIT_COORDINATOR_SPEC.metadata.get("category") == "audit"