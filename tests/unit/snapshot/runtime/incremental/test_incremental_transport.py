# tests/unit/snapshot/runtime/incremental/test_incremental_transport.py

import pytest

from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.serializers import SerializerRegistry
from src.writing.snapshot.runtime.compression import CompressionRegistry
from src.writing.snapshot.runtime.chunking import FixedChunkStrategy
from src.writing.snapshot.runtime.incremental import (
    IncrementalTransport,
    MemoryChunkRepository,
    ChunkSet,
    DeltaChunkSet,
)
from src.writing.snapshot.migration import RawSnapshot, SchemaVersion


class TestIncrementalTransport:
    def setup_method(self):
        self.repo = MemoryChunkRepository()
        self.serializer_registry = SerializerRegistry.with_builtin()
        self.compression_registry = CompressionRegistry.with_builtin()
        self.transport = IncrementalTransport(
            repository=self.repo,
            serializer_resolver=self.serializer_registry,
            compression_resolver=self.compression_registry,
            strategy=FixedChunkStrategy(1024),
        )

    def _assert_snapshot_data(self, snapshot: RawSnapshot, expected_data: dict):
        """辅助方法：验证 RawSnapshot 的数据内容"""
        mapping = snapshot.to_mapping()
        # RawSnapshot.to_mapping() 返回 {"schema_version": ..., "data": ...}
        if "data" in mapping:
            assert mapping["data"] == expected_data
        else:
            # 如果返回的是扁平字典，直接比较
            assert mapping == expected_data

    def test_write_base(self):
        sid = SnapshotId.new()
        snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"test": "data"},
        )
        self.transport.write(sid, snapshot)

        assert self.repo.exists(sid) is True
        loaded = self.repo.load_version(sid)
        assert isinstance(loaded, ChunkSet)
        manifest = self.repo.load_manifest(sid)
        assert manifest.parent_id is None

    def test_write_delta(self):
        base_id = SnapshotId.new()
        base_snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"a": 1},
        )
        self.transport.write(base_id, base_snapshot)

        delta_id = SnapshotId.new()
        delta_snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"a": 1, "b": 2},
        )
        self.transport.write(delta_id, delta_snapshot)

        loaded = self.repo.load_version(delta_id)
        assert isinstance(loaded, DeltaChunkSet)
        manifest = self.repo.load_manifest(delta_id)
        assert manifest.parent_id == base_id
        assert manifest.metadata.get("storage_mode") == "delta"

    def test_read_merges_deltas(self):
        base_id = SnapshotId.new()
        base_snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"a": 1},
        )
        self.transport.write(base_id, base_snapshot)

        delta_id = SnapshotId.new()
        delta_snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"a": 1, "b": 2},
        )
        self.transport.write(delta_id, delta_snapshot)

        restored = self.transport.read(delta_id)
        self._assert_snapshot_data(restored, {"a": 1, "b": 2})

    def test_read_base_directly(self):
        base_id = SnapshotId.new()
        base_snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"x": 10},
        )
        self.transport.write(base_id, base_snapshot)

        restored = self.transport.read(base_id)
        self._assert_snapshot_data(restored, {"x": 10})

    def test_overwrite_existing_version(self):
        sid = SnapshotId.new()
        first = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"v": "first"},
        )
        self.transport.write(sid, first)

        second = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"v": "second"},
        )
        self.transport.write(sid, second)

        restored = self.transport.read(sid)
        self._assert_snapshot_data(restored, {"v": "second"})

    def test_multi_delta_chain(self):
        v1 = SnapshotId.new()
        snapshot_v1 = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"a": 1},
        )
        self.transport.write(v1, snapshot_v1)

        v2 = SnapshotId.new()
        snapshot_v2 = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"a": 1, "b": 2},
        )
        self.transport.write(v2, snapshot_v2)

        v3 = SnapshotId.new()
        snapshot_v3 = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"a": 1, "b": 2, "c": 3},
        )
        self.transport.write(v3, snapshot_v3)

        assert isinstance(self.repo.load_version(v2), DeltaChunkSet)
        assert isinstance(self.repo.load_version(v3), DeltaChunkSet)

        restored = self.transport.read(v3)
        self._assert_snapshot_data(restored, {"a": 1, "b": 2, "c": 3})