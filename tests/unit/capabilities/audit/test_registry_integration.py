# tests/unit/capabilities/audit/test_registry_integration.py

import pytest
from src.capabilities import CapabilityRegistry
from src.capabilities.audit import (
    AUDIT_COORDINATOR_SPEC,
    AUDIT_COORDINATOR_ID,
    AuditCapabilityAdapter,
)


class TestAuditCapabilityRegistryIntegration:

    def test_register_and_require(self):
        registry = CapabilityRegistry()

        cap = AuditCapabilityAdapter.create()
        registry.register(AUDIT_COORDINATOR_SPEC, cap)

        retrieved = registry.require(AUDIT_COORDINATOR_ID)
        assert retrieved is cap
        coordinator = retrieved.get()
        assert coordinator is not None

    def test_require_unknown_raises(self):
        registry = CapabilityRegistry()
        with pytest.raises(KeyError, match="Capability not found"):
            registry.require("unknown.capability")

    def test_register_duplicate_raises(self):
        registry = CapabilityRegistry()
        cap1 = AuditCapabilityAdapter.create()
        cap2 = AuditCapabilityAdapter.create()

        registry.register(AUDIT_COORDINATOR_SPEC, cap1)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(AUDIT_COORDINATOR_SPEC, cap2)

    def test_freeze_prevents_register(self):
        registry = CapabilityRegistry()
        cap = AuditCapabilityAdapter.create()
        registry.register(AUDIT_COORDINATOR_SPEC, cap)
        frozen = registry.freeze()

        with pytest.raises(RuntimeError, match="frozen"):
            registry.register(AUDIT_COORDINATOR_SPEC, cap)

        # frozen registry 仍可查询
        retrieved = frozen.require(AUDIT_COORDINATOR_ID)
        assert retrieved is cap

    def test_full_audit_flow_via_capability(self):
        """
        验证通过 Registry 获取的 Audit 能力能正常执行完整审计。
        """
        registry = CapabilityRegistry()
        cap = AuditCapabilityAdapter.create()
        registry.register(AUDIT_COORDINATOR_SPEC, cap)

        retrieved = registry.require(AUDIT_COORDINATOR_ID)
        coordinator = retrieved.get()

        with coordinator.audit("test-novel", 1, 1, 0) as ctx:
            assert ctx.execution_id is not None
            assert ctx.collector is not None