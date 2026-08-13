# tests/unit/snapshot/runtime/remote/gc/test_retention_reconciler.py

import pytest
from unittest.mock import Mock
from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.remote.gc import (
    ChunkRef,
    RetentionReconciler,
    ChunkExistenceChecker,
    DeletionMarkerScanner,
    DeletionMarkerStore,
    ChunkRetentionChecker,
    RetentionDecision,
    DeletionMarkerError,
)


class TestRetentionReconciler:

    def setup_method(self):
        self.marker_scanner = Mock(spec=DeletionMarkerScanner)
        self.marker_store = Mock(spec=DeletionMarkerStore)
        self.existence_checker = Mock(spec=ChunkExistenceChecker)
        self.retention_checker = Mock(spec=ChunkRetentionChecker)
        self.reconciler = RetentionReconciler(
            marker_scanner=self.marker_scanner,
            marker_store=self.marker_store,
            existence_checker=self.existence_checker,
            retention_checker=self.retention_checker,
        )

    def test_reconcile_clears_protected_markers(self):
        sid = SnapshotId.new()
        markers = [ChunkRef(sid, 1)]
        self.marker_scanner.iter_pending_markers.return_value = iter(markers)
        self.existence_checker.exists.return_value = True
        self.retention_checker.decide.return_value = RetentionDecision.RETAIN
        self.marker_store.clear_marker.return_value = None

        result = self.reconciler.reconcile(auto_clear=True)
        assert result.scanned_markers == 1
        assert result.protected_found == 1
        assert result.protected_cleared == 1
        self.marker_store.clear_marker.assert_called_once()

    def test_reconcile_ignores_stale_markers(self):
        sid = SnapshotId.new()
        markers = [ChunkRef(sid, 1)]
        self.marker_scanner.iter_pending_markers.return_value = iter(markers)
        self.existence_checker.exists.return_value = False
        self.retention_checker.decide.return_value = RetentionDecision.RETAIN

        result = self.reconciler.reconcile(auto_clear=True)
        assert result.protected_found == 0
        assert result.protected_cleared == 0
        self.retention_checker.decide.assert_not_called()

    def test_reconcile_ignores_delete_decision(self):
        sid = SnapshotId.new()
        markers = [ChunkRef(sid, 1)]
        self.marker_scanner.iter_pending_markers.return_value = iter(markers)
        self.existence_checker.exists.return_value = True
        self.retention_checker.decide.return_value = RetentionDecision.DELETE

        result = self.reconciler.reconcile(auto_clear=True)
        assert result.protected_found == 0
        assert result.protected_cleared == 0
        self.marker_store.clear_marker.assert_not_called()

    def test_reconcile_counts_errors(self):
        sid = SnapshotId.new()
        markers = [ChunkRef(sid, 1)]
        self.marker_scanner.iter_pending_markers.return_value = iter(markers)
        self.existence_checker.exists.return_value = True
        self.retention_checker.decide.return_value = RetentionDecision.RETAIN
        self.marker_store.clear_marker.side_effect = DeletionMarkerError("Clear failed")

        result = self.reconciler.reconcile(auto_clear=True)
        assert result.protected_found == 1
        assert result.protected_cleared == 0
        assert result.errors == 1