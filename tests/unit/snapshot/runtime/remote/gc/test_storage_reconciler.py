# tests/unit/snapshot/runtime/remote/gc/test_storage_reconciler.py

import pytest
from unittest.mock import Mock
from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.remote.gc import (
    ChunkRef,
    StorageReconciler,
    ChunkExistenceChecker,
    DeletionMarkerScanner,
    DeletionMarkerStore,
    DeletionMarkerError,
)


class TestStorageReconciler:

    def setup_method(self):
        self.marker_scanner = Mock(spec=DeletionMarkerScanner)
        self.marker_store = Mock(spec=DeletionMarkerStore)
        self.existence_checker = Mock(spec=ChunkExistenceChecker)
        self.reconciler = StorageReconciler(
            marker_scanner=self.marker_scanner,
            marker_store=self.marker_store,
            existence_checker=self.existence_checker,
        )

    def test_reconcile_clears_stale_markers(self):
        sid = SnapshotId.new()
        markers = [ChunkRef(sid, 1)]
        self.marker_scanner.iter_pending_markers.return_value = iter(markers)
        self.existence_checker.exists.return_value = False
        self.marker_store.clear_marker.return_value = None

        result = self.reconciler.reconcile(auto_clear=True)
        assert result.scanned_markers == 1
        assert result.stale_found == 1
        assert result.stale_cleared == 1
        self.marker_store.clear_marker.assert_called_once()

    def test_reconcile_does_not_clear_when_auto_clear_false(self):
        sid = SnapshotId.new()
        markers = [ChunkRef(sid, 1)]
        self.marker_scanner.iter_pending_markers.return_value = iter(markers)
        self.existence_checker.exists.return_value = False

        result = self.reconciler.reconcile(auto_clear=False)
        assert result.stale_found == 1
        assert result.stale_cleared == 0
        self.marker_store.clear_marker.assert_not_called()

    def test_reconcile_counts_errors(self):
        sid = SnapshotId.new()
        markers = [ChunkRef(sid, 1)]
        self.marker_scanner.iter_pending_markers.return_value = iter(markers)
        self.existence_checker.exists.return_value = False
        self.marker_store.clear_marker.side_effect = DeletionMarkerError("Clear failed")

        result = self.reconciler.reconcile(auto_clear=True)
        assert result.stale_found == 1
        assert result.stale_cleared == 0
        assert result.errors == 1