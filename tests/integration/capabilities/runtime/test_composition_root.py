# tests/integration/capabilities/runtime/test_composition_root.py

import pytest
from src.writing.bootstrap.composition_root import build_writer_runtime
from src.writing.bootstrap.snapshot_factory import build_runtime_snapshot
from src.writing.snapshot.runtime.models import RuntimeSnapshot


class TestCompositionRoot:
    def test_build_writer_runtime_succeeds(self):
        runtime = build_writer_runtime()
        assert runtime.runtime_capabilities is not None
        assert runtime.runtime_services is not None

    def test_build_runtime_snapshot_uses_runtime(self):
        runtime = build_writer_runtime()
        snapshot = build_runtime_snapshot(runtime, "novel-123", 1, 1, 0)
        assert isinstance(snapshot, RuntimeSnapshot)
        assert snapshot.runtime_capabilities is runtime.runtime_capabilities

    def test_runtime_snapshot_has_required_capabilities(self):
        runtime = build_writer_runtime()
        snapshot = build_runtime_snapshot(runtime)
        caps = snapshot.runtime_capabilities
        assert caps.has("builtin.runtime.audit.coordinator") is True
        assert caps.has("builtin.runtime.snapshot.repository") is True
        assert caps.has("builtin.runtime.snapshot.version_store") is True
        assert caps.has("builtin.runtime.snapshot.transport") is True

    def test_runtime_services_audit_works_with_snapshot(self):
        runtime = build_writer_runtime()
        with runtime.runtime_services.audit_context("novel", 1, 1, 0) as ctx:
            assert ctx.execution_id is not None
            assert ctx.collector is not None