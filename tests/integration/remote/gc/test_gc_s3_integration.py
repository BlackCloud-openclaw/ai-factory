# tests/integration/remote/gc/test_gc_s3_integration.py
"""
B4.7.1: GC 与 S3 集成测试（使用 moto）
"""

import pytest
import boto3
from moto import mock_aws
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.remote.s3 import (
    S3Client, S3Config, S3KeyLayout, S3ChunkStore, S3VersionStore
)
from src.writing.snapshot.runtime.remote.gc import (
    GarbageCollector,
    S3LeaseManager,
    S3DeletionMarkerStore,
    ChunkRef,
    ChunkMetadata,
)
from src.writing.snapshot.runtime.remote.gc.capability import (
    ChunkEnumerator, GCDeleteAdapter, ChunkMetadataProvider
)
from src.writing.snapshot.runtime.incremental import MemoryVersionStore, VersionManifest
from src.writing.snapshot.runtime.chunking import Chunk


class MockChunk(Chunk):
    def __init__(self, chunk_id: int, payload: bytes):
        super().__init__(chunk_id=chunk_id, payload=payload)


class MockChunkEnumerator(ChunkEnumerator):
    def __init__(self, chunks: list[ChunkRef]):
        self.chunks = chunks

    def list_all_chunks(self):
        return self.chunks

    def list_chunks(self, snapshot_id: SnapshotId):
        return [c for c in self.chunks if c.snapshot_id == snapshot_id]


class MockGCDeleteAdapter(GCDeleteAdapter):
    def __init__(self, client: S3Client, key_layout: S3KeyLayout):
        self.client = client
        self.key_layout = key_layout
        self.deleted = []

    def delete_chunks(self, chunks: list[ChunkRef]):
        keys = [self.key_layout.chunk_key(c.snapshot_id, c.chunk_id) for c in chunks]
        self.client.delete_objects(keys)
        self.deleted.extend(chunks)


class MockChunkMetadataProvider(ChunkMetadataProvider):
    def get_metadata(self, chunk_ref: ChunkRef) -> ChunkMetadata:
        return ChunkMetadata(chunk_ref=chunk_ref, size_bytes=100)


@mock_aws
class TestGCS3Integration:

    def setup_method(self, method=None):
        # 创建 S3 bucket
        self.bucket = "test-gc-bucket"
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=self.bucket)

        self.config = S3Config(bucket=self.bucket, prefix="snapshots/")
        self.client = S3Client(self.config)
        self.key_layout = S3KeyLayout(self.config.prefix)
        self.version_store = MemoryVersionStore()
        self.chunk_store = S3ChunkStore(self.client, self.key_layout)
        self.version_store_s3 = S3VersionStore(self.client, self.key_layout)
        self.lease_manager = S3LeaseManager(self.client, self.key_layout)
        self.deletion_marker_store = S3DeletionMarkerStore(self.client, self.key_layout)

        # 延迟设置 gc，因为需要注入 chunk_enumerator 等
        self.gc = None

    def _create_gc(self, chunk_enumerator, delete_adapter, metadata_provider=None):
        return GarbageCollector(
            version_store=self.version_store,
            chunk_enumerator=chunk_enumerator,
            delete_adapter=delete_adapter,
            metadata_provider=metadata_provider or MockChunkMetadataProvider(),
            lease_manager=self.lease_manager,
            deletion_marker_store=self.deletion_marker_store,
            grace_period_seconds=0,  # 立即删除，测试用
            owner_id="test-owner",
        )

    def test_gc_deletes_orphan_chunks_from_s3(self):
        """孤儿 chunk 被物理删除。"""
        # 创建一个孤儿 chunk（snapshot_id 不在 version_store 中）
        orphan_sid = SnapshotId.new()
        orphan_ref = ChunkRef(orphan_sid, 1)
        self.chunk_store.write_chunk(orphan_sid, MockChunk(1, b"orphan"))

        # 也创建一个可达的 chunk（属于一个有效的 snapshot）
        sid = SnapshotId.new()
        self.version_store.put(VersionManifest(sid, parent_id=None, metadata={}))
        reachable_ref = ChunkRef(sid, 1)
        self.chunk_store.write_chunk(sid, MockChunk(1, b"reachable"))

        all_chunks = [orphan_ref, reachable_ref]
        enumerator = MockChunkEnumerator(all_chunks)
        delete_adapter = MockGCDeleteAdapter(self.client, self.key_layout)

        gc = self._create_gc(enumerator, delete_adapter)
        result = gc.collect(dry_run=False)

        assert result.deleted_count == 1
        assert orphan_ref in result.deleted_chunks
        assert reachable_ref not in result.deleted_chunks

        # 验证 S3 中 orphan chunk 已删除，reachable 还在
        orphan_key = self.key_layout.chunk_key(orphan_sid, 1)
        reachable_key = self.key_layout.chunk_key(sid, 1)
        assert self.client.head_object(orphan_key) is None  # 不存在
        assert self.client.head_object(reachable_key) is not None

    def test_lease_prevents_concurrent_deletion(self):
        """租约防止并发删除同一 chunk。"""
        # 创建一个孤儿 chunk（不属于任何 snapshot）
        orphan_sid = SnapshotId.new()
        orphan_ref = ChunkRef(orphan_sid, 1)
        self.chunk_store.write_chunk(orphan_sid, MockChunk(1, b"orphan"))

        all_chunks = [orphan_ref]
        enumerator = MockChunkEnumerator(all_chunks)
        delete_adapter = MockGCDeleteAdapter(self.client, self.key_layout)

        # 第一个 GC 实例（获取租约并删除）
        gc1 = self._create_gc(enumerator, delete_adapter, metadata_provider=MockChunkMetadataProvider())
        result1 = gc1.collect(dry_run=False)
        assert result1.deleted_count == 1
        assert orphan_ref in result1.deleted_chunks

        # 验证租约已被释放
        assert not self.lease_manager.is_held(orphan_ref)

        # 第二个 GC 实例（此时 chunk 已物理删除，所以不会删除任何东西）
        # 为了真正模拟并发，我们应模拟两个实例同时竞争，但这里我们只能顺序测试。
        # 我们重新创建一个同名的 chunk 来模拟另一个实例看到同一个孤儿 chunk 但尚未被删除。
        # 更真实的测试：在第一个 GC 删除前，模拟第二个实例尝试获取租约。
        # 由于 moto 不能轻易模拟并发，我们调整测试：模拟两个 GC 实例同时尝试获取租约，
        # 可以通过直接调用 lease_manager.acquire 来模拟。
        # 但这里我们测试的是租约防止并发删除：当租约被持有时，另一个实例不能删除。
        # 为此，我们先手动获取租约，然后运行 GC，应该不会删除。
        # 更简单：我们直接验证租约被持有期间，GC 无法删除。
        self.chunk_store.write_chunk(orphan_sid, MockChunk(1, b"orphan2"))  # 重新写入

        # 手动获取租约（模拟另一个节点持有）
        acquired = self.lease_manager.acquire(orphan_ref, 60, "other-node")
        assert acquired is True

        # 第二个 GC 实例尝试删除（此时租约被 other-node 持有，应该跳过）
        gc2 = self._create_gc(enumerator, delete_adapter, metadata_provider=MockChunkMetadataProvider())
        result2 = gc2.collect(dry_run=False)
        # 因为租约被持有，GC 应跳过该 chunk，所以删除数为 0
        assert result2.deleted_count == 0
        assert orphan_ref not in result2.deleted_chunks

        # 释放租约，然后 GC 应该能删除
        self.lease_manager.release(orphan_ref, "other-node")
        gc3 = self._create_gc(enumerator, delete_adapter, metadata_provider=MockChunkMetadataProvider())
        result3 = gc3.collect(dry_run=False)
        assert result3.deleted_count == 1
        assert orphan_ref in result3.deleted_chunks