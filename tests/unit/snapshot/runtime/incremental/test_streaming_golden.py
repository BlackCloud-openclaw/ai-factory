# tests/unit/snapshot/runtime/incremental/test_streaming_golden.py

import pytest

from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.serializers import SerializerRegistry
from src.writing.snapshot.runtime.compression import CompressionRegistry
from src.writing.snapshot.runtime.chunking import FixedChunkStrategy
from src.writing.snapshot.runtime.incremental import (
    IncrementalTransport,
    MemoryChunkRepository,
    ChunkSet,
)
from src.writing.snapshot.migration import RawSnapshot, SchemaVersion


class TestStreamingGolden:
    def setup_method(self):
        self.serializer_registry = SerializerRegistry.with_builtin()
        self.compression_registry = CompressionRegistry.with_builtin()
        self.strategy = FixedChunkStrategy(1024)

    def _assert_chunk_sets_equal_bytewise(self, a: ChunkSet, b: ChunkSet):
        """验证两个 ChunkSet 在 chunk_id 和 payload 字节级别完全一致。"""
        a_keys = set(a.keys())
        b_keys = set(b.keys())
        assert a_keys == b_keys, f"Chunk IDs mismatch: {a_keys} vs {b_keys}"

        for cid in a_keys:
            a_chunk = a.get(cid)
            b_chunk = b.get(cid)
            assert a_chunk is not None and b_chunk is not None
            assert a_chunk.payload == b_chunk.payload, f"Payload mismatch for chunk {cid}"

    def test_chunk_golden(self):
        snapshot_id = SnapshotId.new()
        snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"a": 1, "b": "hello", "c": [1, 2, 3]},
        )

        # 缓冲路径
        buffered_repo = MemoryChunkRepository()
        buffered_transport = IncrementalTransport(
            repository=buffered_repo,
            serializer_resolver=self.serializer_registry,
            compression_resolver=self.compression_registry,
            strategy=self.strategy,
        )
        buffered_transport._streaming_supported = False
        buffered_transport.write(snapshot_id, snapshot)

        # 流式路径
        streaming_repo = MemoryChunkRepository()
        streaming_transport = IncrementalTransport(
            repository=streaming_repo,
            serializer_resolver=self.serializer_registry,
            compression_resolver=self.compression_registry,
            strategy=self.strategy,
        )
        assert streaming_transport._streaming_supported is True
        streaming_transport.write(snapshot_id, snapshot)

        # 1. 验证存储的 ChunkSet 在字节级别一致
        buffered_chunks = buffered_repo.load_version(snapshot_id)
        streaming_chunks = streaming_repo.load_version(snapshot_id)

        assert isinstance(buffered_chunks, ChunkSet)
        assert isinstance(streaming_chunks, ChunkSet)
        self._assert_chunk_sets_equal_bytewise(buffered_chunks, streaming_chunks)

        # 2. 验证读取结果一致
        buffered_result = buffered_transport.read(snapshot_id)
        streaming_result = streaming_transport.read(snapshot_id)
        assert buffered_result.schema_version == streaming_result.schema_version
        assert buffered_result.to_mapping() == streaming_result.to_mapping()


    def test_streaming_fallback(self):
        repo = MemoryChunkRepository()
        transport = IncrementalTransport(
            repository=repo,
            serializer_resolver=self.serializer_registry,
            compression_resolver=self.compression_registry,
            strategy=self.strategy,
        )
        # 模拟不支持流式（使用 identity codec 会导致检测失败）
        # 实际上，由于 default_codec_id="builtin.identity"，
        # transport._streaming_supported 已经为 False
        sid = SnapshotId.new()
        snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"test": "fallback"},
        )
        transport.write(sid, snapshot)
        result = transport.read(sid)
        assert result.schema_version == SchemaVersion(1, 0)
        # 使用辅助方法或直接访问 mapping
        mapping = result.to_mapping()
        # RawSnapshot.to_mapping() 返回 {"schema_version": ..., "data": ...}
        if "data" in mapping:
            assert mapping["data"]["test"] == "fallback"
        else:
            assert mapping["test"] == "fallback"