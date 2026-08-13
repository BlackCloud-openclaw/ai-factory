"""
B4.7.1: GarbageCollector 安全删除逻辑测试（租约、标记、Grace Period）
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta, timezone

from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.incremental import MemoryVersionStore, VersionManifest
from src.writing.snapshot.runtime.remote.gc import (
    GarbageCollector,
    ChunkRef,
    ChunkMetadata,
    GCNotSupportedError,
    GCResult,
    DeletionCandidate,
    DeletionPlan,
    LeaseManager,
    DeletionMarkerStore,
    LeaseConflictError,
    LeaseAcquisitionError,
    GracePeriodNotElapsedError,
    DeletionFailedError,
)
from src.writing.snapshot.runtime.remote.gc.capability import (
    ChunkEnumerator,
    GCDeleteAdapter,
    ChunkMetadataProvider,
)


class TestGarbageCollectorSecurity:

    def setup_method(self):
        self.version_store = MemoryVersionStore()
        self.chunk_enumerator = Mock(spec=ChunkEnumerator)
        self.delete_adapter = Mock(spec=GCDeleteAdapter)
        self.metadata_provider = Mock(spec=ChunkMetadataProvider)
        self.lease_manager = Mock(spec=LeaseManager)
        self.deletion_marker_store = Mock(spec=DeletionMarkerStore)

        self.chunk_enumerator.list_chunks.return_value = []
        self.chunk_enumerator.list_all_chunks.return_value = []
        self.delete_adapter.delete_chunks.return_value = None

        # 默认租约成功
        self.lease_manager.acquire.return_value = True
        self.lease_manager.release.return_value = None

        # 默认标记存储：未标记，grace period ready
        self.deletion_marker_store.get_deletion_info.return_value = None
        self.deletion_marker_store.is_ready_for_physical_deletion.return_value = True
        self.deletion_marker_store.mark_for_deletion.return_value = None
        self.deletion_marker_store.clear_marker.return_value = None

        self.gc = GarbageCollector(
            version_store=self.version_store,
            chunk_enumerator=self.chunk_enumerator,
            delete_adapter=self.delete_adapter,
            metadata_provider=self.metadata_provider,
            lease_manager=self.lease_manager,
            deletion_marker_store=self.deletion_marker_store,
            grace_period_seconds=86400,
            lease_ttl_seconds=300,
            owner_id="test-owner",
        )

    def _add_manifest(self, sid: SnapshotId, parent: SnapshotId | None = None):
        manifest = VersionManifest(snapshot_id=sid, parent_id=parent, metadata={})
        self.version_store.put(manifest)

    def _setup_chunks(self, reachable: list[ChunkRef], physical: list[ChunkRef]):
        """设置可达和物理 chunk 列表。"""
        self.chunk_enumerator.list_all_chunks.return_value = physical
        # 模拟 list_chunks：为每个 reachable 的 snapshot 返回其 chunks
        def list_chunks_side_effect(sid):
            return [c for c in reachable if c.snapshot_id == sid]
        self.chunk_enumerator.list_chunks.side_effect = list_chunks_side_effect

        def metadata_side_effect(ref):
            return ChunkMetadata(ref, size_bytes=1024)
        self.metadata_provider.get_metadata.side_effect = metadata_side_effect

    def test_safe_deletion_skips_chunk_with_lease_conflict(self):
        """租约冲突（被其他节点持有）的 chunk 被跳过。"""
        sid = SnapshotId.new()
        self._add_manifest(sid)
        orphan_sid = SnapshotId.new()
        orphan_ref = ChunkRef(orphan_sid, 1)
        self._setup_chunks([ChunkRef(sid, 1)], [ChunkRef(sid, 1), orphan_ref])

        self.lease_manager.acquire.side_effect = LeaseConflictError("Held by other")

        result = self.gc.collect(dry_run=False)
        # 只有 orphan 应该被删除，但 lease 失败导致无删除
        assert result.deleted_count == 0
        self.delete_adapter.delete_chunks.assert_not_called()
        self.deletion_marker_store.mark_for_deletion.assert_not_called()

    def test_safe_deletion_aborts_on_lease_system_error(self):
        """租约系统错误导致 GC 中止。"""
        sid = SnapshotId.new()
        self._add_manifest(sid)
        orphan_ref = ChunkRef(SnapshotId.new(), 1)
        self._setup_chunks([ChunkRef(sid, 1)], [ChunkRef(sid, 1), orphan_ref])

        self.lease_manager.acquire.side_effect = LeaseAcquisitionError("Network error")

        with pytest.raises(LeaseAcquisitionError):
            self.gc.collect(dry_run=False)

        self.delete_adapter.delete_chunks.assert_not_called()

    def test_safe_deletion_skips_chunk_not_ready_grace_period(self):
        """Grace period 未到，chunk 被跳过。"""
        sid = SnapshotId.new()
        self._add_manifest(sid)
        orphan_ref = ChunkRef(SnapshotId.new(), 1)
        self._setup_chunks([ChunkRef(sid, 1)], [ChunkRef(sid, 1), orphan_ref])

        # 第一次调用：标记删除，然后 is_ready 抛出 GracePeriodNotElapsedError
        self.deletion_marker_store.get_deletion_info.return_value = None
        self.deletion_marker_store.is_ready_for_physical_deletion.side_effect = GracePeriodNotElapsedError("Not ready")

        result = self.gc.collect(dry_run=False)
        assert result.deleted_count == 0
        self.deletion_marker_store.mark_for_deletion.assert_called_once()
        self.delete_adapter.delete_chunks.assert_not_called()

    def test_safe_deletion_proceeds_when_ready(self):
        """所有安全检查通过，正常删除。"""
        sid = SnapshotId.new()
        self._add_manifest(sid)
        orphan_ref = ChunkRef(SnapshotId.new(), 1)
        self._setup_chunks([ChunkRef(sid, 1)], [ChunkRef(sid, 1), orphan_ref])

        # 标记存储：第一次 get_deletion_info 返回 None，触发标记；is_ready 返回 True
        self.deletion_marker_store.get_deletion_info.return_value = None
        self.deletion_marker_store.is_ready_for_physical_deletion.return_value = True

        result = self.gc.collect(dry_run=False)
        assert result.deleted_count == 1
        self.delete_adapter.delete_chunks.assert_called_once_with([orphan_ref])
        self.deletion_marker_store.mark_for_deletion.assert_called_once()
        self.deletion_marker_store.clear_marker.assert_called_once_with(orphan_ref)
        self.lease_manager.acquire.assert_called_once_with(orphan_ref, 300, "test-owner")
        # release 在 finally 中调用一次
        self.lease_manager.release.assert_called_once_with(orphan_ref, "test-owner")

    def test_safe_deletion_batch_delete(self):
        """批处理删除多个 chunk。"""
        sid = SnapshotId.new()
        self._add_manifest(sid)
        orphan_sid = SnapshotId.new()
        orphan_refs = [ChunkRef(orphan_sid, i) for i in range(3)]
        all_physical = [ChunkRef(sid, 1)] + orphan_refs
        self._setup_chunks([ChunkRef(sid, 1)], all_physical)

        self.gc = GarbageCollector(
            version_store=self.version_store,
            chunk_enumerator=self.chunk_enumerator,
            delete_adapter=self.delete_adapter,
            metadata_provider=self.metadata_provider,
            lease_manager=self.lease_manager,
            deletion_marker_store=self.deletion_marker_store,
            batch_size=2,  # 分批大小 2
        )
        result = self.gc.collect(dry_run=False)
        assert result.deleted_count == 3
        # delete_adapter 应被调用两次：第一次2个，第二次1个
        assert self.delete_adapter.delete_chunks.call_count == 2
        calls = self.delete_adapter.delete_chunks.call_args_list
        batch1 = calls[0][0][0]
        batch2 = calls[1][0][0]
        assert len(batch1) == 2
        assert len(batch2) == 1
        assert set(batch1 + batch2) == set(orphan_refs)

    def test_safe_deletion_clears_marker_after_delete(self):
        """删除后清除标记。"""
        sid = SnapshotId.new()
        self._add_manifest(sid)
        orphan_ref = ChunkRef(SnapshotId.new(), 1)
        self._setup_chunks([ChunkRef(sid, 1)], [ChunkRef(sid, 1), orphan_ref])

        self.deletion_marker_store.get_deletion_info.return_value = {
            "delete_time": datetime.now(timezone.utc).isoformat(),
            "grace_period_seconds": 0,
        }
        self.deletion_marker_store.is_ready_for_physical_deletion.return_value = True

        result = self.gc.collect(dry_run=False)
        self.deletion_marker_store.clear_marker.assert_called_once_with(orphan_ref)

    def test_safe_deletion_failure_raises_and_does_not_clear_marker(self):
        """删除失败时抛出异常，不清除标记。"""
        sid = SnapshotId.new()
        self._add_manifest(sid)
        orphan_ref = ChunkRef(SnapshotId.new(), 1)
        self._setup_chunks([ChunkRef(sid, 1)], [ChunkRef(sid, 1), orphan_ref])

        self.delete_adapter.delete_chunks.side_effect = Exception("S3 delete error")

        with pytest.raises(DeletionFailedError):
            self.gc.collect(dry_run=False)

        self.deletion_marker_store.clear_marker.assert_not_called()
        # release 在 finally 中执行一次
        self.lease_manager.release.assert_called_once_with(orphan_ref, "test-owner")

    def test_dry_run_skips_all_safety_checks(self):
        """dry_run 模式跳过所有安全检查和删除。"""
        sid = SnapshotId.new()
        self._add_manifest(sid)
        orphan_ref = ChunkRef(SnapshotId.new(), 1)
        self._setup_chunks([ChunkRef(sid, 1)], [ChunkRef(sid, 1), orphan_ref])

        result = self.gc.collect(dry_run=True)
        assert result.dry_run is True
        assert result.deleted_count == 1
        self.lease_manager.acquire.assert_not_called()
        self.deletion_marker_store.mark_for_deletion.assert_not_called()
        self.delete_adapter.delete_chunks.assert_not_called()

    def test_force_mode_skips_all_safety_checks(self):
        """force=True 跳过所有安全检查。"""
        sid = SnapshotId.new()
        self._add_manifest(sid)
        orphan_ref = ChunkRef(SnapshotId.new(), 1)
        self._setup_chunks([ChunkRef(sid, 1)], [ChunkRef(sid, 1), orphan_ref])

        self.gc.collect(dry_run=False, force=True)
        self.lease_manager.acquire.assert_not_called()
        self.deletion_marker_store.mark_for_deletion.assert_not_called()
        self.deletion_marker_store.is_ready_for_physical_deletion.assert_not_called()
        self.delete_adapter.delete_chunks.assert_called_once_with([orphan_ref])