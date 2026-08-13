# tests/unit/snapshot/runtime/remote/gc/test_gc_collector.py

import pytest
from unittest.mock import Mock

from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.incremental import MemoryVersionStore, VersionManifest
from src.writing.snapshot.runtime.remote.gc import (
    GarbageCollector,
    ChunkRef,
    ChunkMetadata,
    GCNotSupportedError,
    ChunkEnumerator,
    GCDeleteAdapter,
    ChunkMetadataProvider,
    GCInconsistentError,
)


class TestGarbageCollector:
    def setup_method(self):
        self.version_store = MemoryVersionStore()
        self.chunk_enumerator = Mock(spec=ChunkEnumerator)
        self.delete_adapter = Mock(spec=GCDeleteAdapter)
        self.metadata_provider = Mock(spec=ChunkMetadataProvider)

        # 新增：累积 sid -> refs 映射
        self._chunk_map = {}

        self.chunk_enumerator.list_chunks.return_value = []
        self.chunk_enumerator.list_all_chunks.return_value = []
        self.delete_adapter.delete_chunks = Mock(return_value=None)

    def _create_manifest(self, sid: SnapshotId, parent_id: SnapshotId | None = None):
        manifest = VersionManifest(
            snapshot_id=sid,
            parent_id=parent_id,
            metadata={},
        )
        self.version_store.put(manifest)

    def _mock_chunks_for_snapshot(self, sid: SnapshotId, *chunk_ids: int):
        refs = [ChunkRef(sid, cid) for cid in chunk_ids]
        self._chunk_map[sid] = refs   # 累加

        def list_chunks_side_effect(s):
            return self._chunk_map.get(s, [])

        self.chunk_enumerator.list_chunks.side_effect = list_chunks_side_effect
        return refs

    def _mock_all_physical_chunks(self, *refs: ChunkRef):
        self.chunk_enumerator.list_all_chunks.return_value = refs
        return refs

    def test_dry_run_with_single_snapshot(self):
        sid = SnapshotId.new()
        self._create_manifest(sid)
        self._mock_chunks_for_snapshot(sid, 1, 2)

        gc = GarbageCollector(
            self.version_store,
            self.chunk_enumerator,
            self.delete_adapter,
            self.metadata_provider,
        )
        result = gc.collect(dry_run=True)

        assert result.deleted_count == 0
        assert result.dry_run is True

    def test_dry_run_with_orphan_chunks_and_metadata(self):
        sid1 = SnapshotId.new()
        sid2 = SnapshotId.new()
        self._create_manifest(sid1)
        self._create_manifest(sid2, parent_id=sid1)

        self._mock_chunks_for_snapshot(sid1, 1, 2)
        self._mock_chunks_for_snapshot(sid2, 3)

        orphan_sid = SnapshotId.new()
        orphan_ref1 = ChunkRef(orphan_sid, 10)
        orphan_ref2 = ChunkRef(orphan_sid, 11)
        self._mock_all_physical_chunks(
            ChunkRef(sid1, 1), ChunkRef(sid1, 2),
            ChunkRef(sid2, 3),
            orphan_ref1, orphan_ref2,
        )

        def get_metadata_side_effect(ref):
            if ref == orphan_ref1:
                return ChunkMetadata(ref, size_bytes=100)
            if ref == orphan_ref2:
                return ChunkMetadata(ref, size_bytes=200)
            return ChunkMetadata(ref, size_bytes=1024)

        self.metadata_provider.get_metadata.side_effect = get_metadata_side_effect

        gc = GarbageCollector(
            self.version_store,
            self.chunk_enumerator,
            self.delete_adapter,
            self.metadata_provider,
        )
        result = gc.collect(dry_run=True)

        assert orphan_ref1 in result.deleted_chunks
        assert orphan_ref2 in result.deleted_chunks
        assert result.deleted_count == 2
        assert result.reclaimed_bytes == 300

    def test_broken_chain_detection(self):
        a = SnapshotId.new()
        b = SnapshotId.new()
        c = SnapshotId.new()
        self._create_manifest(a, parent_id=None)
        self._create_manifest(c, parent_id=b)

        self._mock_chunks_for_snapshot(a, 1)
        self._mock_chunks_for_snapshot(c, 2)

        orphan_sid = SnapshotId.new()
        self._mock_all_physical_chunks(
            ChunkRef(a, 1),
            ChunkRef(c, 2),
            ChunkRef(orphan_sid, 99),
        )

        gc = GarbageCollector(
            self.version_store,
            self.chunk_enumerator,
            self.delete_adapter,
            self.metadata_provider,
        )

        with pytest.raises(GCInconsistentError, match="Manifest for .* not found"):
            gc.collect(dry_run=True)

    def test_actual_deletion_with_metadata(self):
        sid = SnapshotId.new()
        self._create_manifest(sid)
        self._mock_chunks_for_snapshot(sid, 1)

        orphan_sid = SnapshotId.new()
        orphan_ref = ChunkRef(orphan_sid, 10)
        self._mock_all_physical_chunks(
            ChunkRef(sid, 1),
            orphan_ref,
        )

        def get_metadata_side_effect(ref):
            if ref == orphan_ref:
                return ChunkMetadata(ref, size_bytes=512)
            return ChunkMetadata(ref, size_bytes=1024)

        self.metadata_provider.get_metadata.side_effect = get_metadata_side_effect

        gc = GarbageCollector(
            self.version_store,
            self.chunk_enumerator,
            self.delete_adapter,
            self.metadata_provider,
        )
        result = gc.collect(dry_run=False)

        self.delete_adapter.delete_chunks.assert_called()
        assert orphan_ref in result.deleted_chunks
        assert result.reclaimed_bytes == 512
        assert result.dry_run is False

    def test_metadata_failure_fallback(self):
        sid = SnapshotId.new()
        self._create_manifest(sid)
        self._mock_chunks_for_snapshot(sid, 1)

        orphan_ref = ChunkRef(SnapshotId.new(), 99)
        self._mock_all_physical_chunks(
            ChunkRef(sid, 1),
            orphan_ref,
        )

        self.metadata_provider.get_metadata.side_effect = ValueError("Chunk not found")

        gc = GarbageCollector(
            self.version_store,
            self.chunk_enumerator,
            self.delete_adapter,
            self.metadata_provider,
        )
        result = gc.collect(dry_run=True)

        assert orphan_ref in result.deleted_chunks
        assert result.reclaimed_bytes == 0

    def test_multiple_orphan_metadata_aggregation(self):
        sid = SnapshotId.new()
        self._create_manifest(sid)
        self._mock_chunks_for_snapshot(sid, 1)

        orphan_refs = [
            ChunkRef(SnapshotId.new(), 10),
            ChunkRef(SnapshotId.new(), 11),
            ChunkRef(SnapshotId.new(), 12),
        ]
        all_chunks = [ChunkRef(sid, 1)] + orphan_refs
        self._mock_all_physical_chunks(*all_chunks)

        def get_metadata_side_effect(ref):
            if ref == orphan_refs[0]:
                return ChunkMetadata(ref, size_bytes=100)
            if ref == orphan_refs[1]:
                return ChunkMetadata(ref, size_bytes=200)
            if ref == orphan_refs[2]:
                return ChunkMetadata(ref, size_bytes=300)
            return ChunkMetadata(ref, size_bytes=1024)

        self.metadata_provider.get_metadata.side_effect = get_metadata_side_effect

        gc = GarbageCollector(
            self.version_store,
            self.chunk_enumerator,
            self.delete_adapter,
            self.metadata_provider,
        )
        result = gc.collect(dry_run=True)

        assert len(result.deleted_chunks) == 3
        assert result.reclaimed_bytes == 600

    def test_stats_tracking(self):
        sid = SnapshotId.new()
        self._create_manifest(sid)
        self._mock_chunks_for_snapshot(sid, 1)

        gc = GarbageCollector(
            self.version_store,
            self.chunk_enumerator,
            self.delete_adapter,
            self.metadata_provider,
        )

        gc.collect(dry_run=True)
        stats = gc.stats()
        assert stats.total_runs == 1
        assert stats.last_run is not None

        gc.collect(dry_run=True)
        stats = gc.stats()
        assert stats.total_runs == 2

    def test_reset_stats(self):
        sid = SnapshotId.new()
        self._create_manifest(sid)
        self._mock_chunks_for_snapshot(sid, 1)

        gc = GarbageCollector(
            self.version_store,
            self.chunk_enumerator,
            self.delete_adapter,
            self.metadata_provider,
        )
        gc.collect(dry_run=True)
        assert gc.stats().total_runs == 1

        gc.reset_stats()
        assert gc.stats().total_runs == 0

    def test_metadata_provider_optional(self):
        sid = SnapshotId.new()
        self._create_manifest(sid)
        self._mock_chunks_for_snapshot(sid, 1, 2)

        orphan_ref = ChunkRef(SnapshotId.new(), 99)
        self._mock_all_physical_chunks(
            ChunkRef(sid, 1),
            ChunkRef(sid, 2),
            orphan_ref,
        )

        gc = GarbageCollector(
            self.version_store,
            self.chunk_enumerator,
            self.delete_adapter,
            metadata_provider=None,
        )
        result = gc.collect(dry_run=True)

        assert orphan_ref in result.deleted_chunks
        assert result.reclaimed_bytes == 0