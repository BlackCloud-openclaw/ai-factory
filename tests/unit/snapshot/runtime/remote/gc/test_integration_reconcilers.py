# tests/unit/snapshot/runtime/remote/gc/test_integration_reconcilers.py
"""
B4.10: StorageReconciler + RetentionReconciler 联合使用场景
"""

import pytest
from unittest.mock import Mock
from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.remote.gc import (
    ChunkRef,
    StorageReconciler,
    RetentionReconciler,
    ChunkExistenceChecker,
    DeletionMarkerScanner,
    DeletionMarkerStore,
    ChunkRetentionChecker,
    RetentionDecision,
)


class TestIntegrationReconcilers:

    def test_both_reconcilers_can_run_sequentially(self):
        """存储协调器 + 保留协调器可以串联运行。"""
        sid = SnapshotId.new()
        stale_marker = ChunkRef(sid, 1)
        protected_marker = ChunkRef(sid, 2)

        marker_scanner = Mock(spec=DeletionMarkerScanner)
        marker_scanner.iter_pending_markers.return_value = iter([stale_marker, protected_marker])

        marker_store = Mock(spec=DeletionMarkerStore)
        marker_store.clear_marker.return_value = None

        existence_checker = Mock(spec=ChunkExistenceChecker)
        # stale_marker: chunk 不存在，protected_marker: chunk 存在
        def fake_exists(ref):
            return ref == protected_marker
        existence_checker.exists.side_effect = fake_exists

        retention_checker = Mock(spec=ChunkRetentionChecker)
        retention_checker.decide.return_value = RetentionDecision.RETAIN

        # 1. StorageReconciler
        storage_reconciler = StorageReconciler(
            marker_scanner=marker_scanner,
            marker_store=marker_store,
            existence_checker=existence_checker,
        )
        storage_result = storage_reconciler.reconcile(auto_clear=True)

        # 2. RetentionReconciler（同一个 scanner，会重新扫描）
        # 注意：marker_scanner.iter_pending_markers 已被消费完，需要重新设置
        marker_scanner.iter_pending_markers.return_value = iter([protected_marker])

        retention_reconciler = RetentionReconciler(
            marker_scanner=marker_scanner,
            marker_store=marker_store,
            existence_checker=existence_checker,
            retention_checker=retention_checker,
        )
        retention_result = retention_reconciler.reconcile(auto_clear=True)

        assert storage_result.stale_found == 1
        assert storage_result.stale_cleared == 1

        assert retention_result.protected_found == 1
        assert retention_result.protected_cleared == 1

        # 总共清除 2 个 marker
        assert marker_store.clear_marker.call_count == 2