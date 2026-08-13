# tests/integration/capabilities/runtime/test_audit_runtime_registration.py

import pytest
from src.capabilities.runtime import RuntimeCapabilityRegistry
from src.capabilities.audit import (
    AUDIT_COORDINATOR_ID,
    AUDIT_COORDINATOR_SPEC,
    AuditCapabilityAdapter,
)
from src.writing.audit import AuditCoordinator
from src.writing.bootstrap.runtime_capabilities import (
    build_runtime_capabilities,
)


class TestAuditRuntimeRegistration:

    def test_audit_capability_registered(self):
        """验证 AuditCapability 已注册到 Runtime Registry。"""
        registry = build_runtime_capabilities()
        assert registry.has(AUDIT_COORDINATOR_ID) is True
        ids = registry.list_ids()
        assert AUDIT_COORDINATOR_ID in ids

    def test_audit_capability_can_be_retrieved(self):
        """验证通过 Registry 可以获取 AuditCapability。"""
        registry = build_runtime_capabilities()
        capability = registry.require(AUDIT_COORDINATOR_ID)
        assert capability is not None
        assert hasattr(capability, "get")
        assert callable(capability.get)

    def test_audit_capability_returns_audit_coordinator(self):
        """验证 capability.get() 返回 AuditCoordinator 实例。"""
        registry = build_runtime_capabilities()
        capability = registry.require(AUDIT_COORDINATOR_ID)
        coordinator = capability.get()
        assert isinstance(coordinator, AuditCoordinator)
        assert hasattr(coordinator, "start")
        assert hasattr(coordinator, "audit")

    def test_manual_registration_works(self):
        """验证手动注册流程（不使用 bootstrap）。"""
        registry = RuntimeCapabilityRegistry()
        capability = AuditCapabilityAdapter.create()
        registry.register(AUDIT_COORDINATOR_SPEC, capability)
        registry.freeze()
        retrieved = registry.require(AUDIT_COORDINATOR_ID)
        coordinator = retrieved.get()
        assert isinstance(coordinator, AuditCoordinator)

    def test_runtime_snapshot_has_capabilities_field(self):
        """验证 RuntimeSnapshot 有 runtime_capabilities 字段。"""
        from src.writing.snapshot.runtime.models import RuntimeSnapshot
        from src.writing.snapshot.runtime.id import SnapshotId
        registry = build_runtime_capabilities()
        snapshot = RuntimeSnapshot(
            identity=SnapshotId.new(),
            manifest=None,
            metadata=None,
            planning=None,
            writer_ir=None,
            prompt_bundle=None,
            render_trace=None,
            draft="",
            coverage=None,
            timestamp=None,
            runtime_capabilities=registry,
        )
        assert hasattr(snapshot, "runtime_capabilities")
        assert snapshot.runtime_capabilities is not None
        assert snapshot.runtime_capabilities.has(AUDIT_COORDINATOR_ID) is True