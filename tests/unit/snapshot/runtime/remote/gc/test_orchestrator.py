# tests/unit/snapshot/runtime/remote/gc/test_orchestrator.py

import pytest
from unittest.mock import Mock

from src.writing.snapshot.runtime.remote.gc import (
    GCOrchestrator,
    OrchestratorConfig,
    StorageReconciler,
    RetentionReconciler,
    GarbageCollector,
    LeaseManager,
    GarbageCollectionError,
    GCResult,
    MarkerReconciliationResult,
)


class TestGCOrchestrator:

    def setup_method(self):
        self.storage_reconciler = Mock(spec=StorageReconciler)
        self.retention_reconciler = Mock(spec=RetentionReconciler)
        self.garbage_collector = Mock(spec=GarbageCollector)
        self.lease_manager = Mock(spec=LeaseManager)
        self.config = OrchestratorConfig(dry_run=True)

        # 默认行为
        self.storage_reconciler.reconcile.return_value = MarkerReconciliationResult()
        self.retention_reconciler.reconcile.return_value = MarkerReconciliationResult()
        self.garbage_collector.collect.return_value = GCResult(
            deleted_chunks=frozenset(),
            reclaimed_bytes=0,
            dry_run=True,
        )

        # Lease scope 上下文管理器
        self._scope_ctx = Mock()
        self._scope_ctx.__enter__ = Mock(return_value=None)
        self._scope_ctx.__exit__ = Mock(return_value=False)
        self.lease_manager.scope.return_value = self._scope_ctx

        self.orchestrator = GCOrchestrator(
            storage_reconciler=self.storage_reconciler,
            retention_reconciler=self.retention_reconciler,
            garbage_collector=self.garbage_collector,
            lease_manager=self.lease_manager,
            config=self.config,
        )

    def test_orchestrator_acquires_lease_via_context_manager(self):
        result = self.orchestrator.run()
        assert result.success
        self.lease_manager.scope.assert_called_once()
        self._scope_ctx.__enter__.assert_called_once()
        self._scope_ctx.__exit__.assert_called_once()

    def test_orchestrator_calls_storage_reconcile(self):
        result = self.orchestrator.run()
        self.storage_reconciler.reconcile.assert_called_once()

    def test_orchestrator_calls_retention_reconcile(self):
        result = self.orchestrator.run()
        self.retention_reconciler.reconcile.assert_called_once()

    def test_orchestrator_calls_garbage_collector(self):
        result = self.orchestrator.run()
        self.garbage_collector.collect.assert_called_once_with(dry_run=True)

    def test_orchestrator_on_lease_failure(self):
        self.lease_manager.scope.side_effect = GarbageCollectionError("Failed to acquire")
        with pytest.raises(GarbageCollectionError, match="Failed to acquire"):
            self.orchestrator.run()

    def test_orchestrator_propagates_collector_exceptions(self):
        self.garbage_collector.collect.side_effect = RuntimeError("Collector failed")
        with pytest.raises(RuntimeError, match="Collector failed"):
            self.orchestrator.run()

    def test_orchestrator_result_summary(self):
        self.garbage_collector.collect.return_value = GCResult(
            deleted_chunks=frozenset(),
            reclaimed_bytes=1024,
            dry_run=False,
        )
        result = self.orchestrator.run()
        assert result.deleted_count == 0
        assert result.reclaimed_bytes == 1024

    def test_run_reconciliation(self):
        storage_result, retention_result = self.orchestrator.run_reconciliation()
        self.storage_reconciler.reconcile.assert_called_once()
        self.retention_reconciler.reconcile.assert_called_once()
        assert storage_result is not None
        assert retention_result is not None

    def test_reconciliation_only_skips_gc(self):
        self.orchestrator.run_reconciliation()
        self.garbage_collector.collect.assert_not_called()