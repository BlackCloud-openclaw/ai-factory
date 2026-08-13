# tests/unit/snapshot/runtime/test_compression.py

import pytest

from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.compression import (
    CompressionRegistry,
    IdentityCodec,
    GzipCodec,
    DuplicateCompressionCodecError,
    UnsupportedCompressionError,
    CompressionDataError,
)
from src.writing.snapshot.runtime import RuntimePipeline, RecordBuilder
from src.writing.snapshot.runtime.memory_store import MemorySnapshotStore
from src.writing.snapshot.runtime.serializers import JsonSerializer, SerializerRegistry
from src.writing.snapshot.migration import RawSnapshot, SchemaVersion


class TestIdentityCodec:
    def test_compress_identity(self):
        codec = IdentityCodec()
        data = b"hello"
        assert codec.compress(data) == data
        assert codec.decompress(data) == data

    def test_id(self):
        assert IdentityCodec().id == "builtin.identity"


class TestGzipCodec:
    def test_round_trip(self):
        codec = GzipCodec()
        original = b"hello world" * 100
        compressed = codec.compress(original)
        decompressed = codec.decompress(compressed)
        assert decompressed == original
        assert len(compressed) < len(original)

    def test_deterministic_output(self):
        codec = GzipCodec()
        data = b"test data" * 10
        assert codec.compress(data) == codec.compress(data)

    def test_id(self):
        assert GzipCodec().id == "builtin.gzip"

    def test_bad_gzip_data_raises(self):
        codec = GzipCodec()
        with pytest.raises(CompressionDataError, match="Gzip decompression failed"):
            codec.decompress(b"not gzip data")


class TestCompressionRegistry:
    def test_register_and_resolve(self):
        registry = CompressionRegistry([IdentityCodec(), GzipCodec()])
        assert registry.resolve("builtin.identity").id == "builtin.identity"
        assert registry.resolve("builtin.gzip").id == "builtin.gzip"
        assert set(registry.list()) == {"builtin.identity", "builtin.gzip"}

    def test_duplicate_raises(self):
        with pytest.raises(DuplicateCompressionCodecError):
            CompressionRegistry([IdentityCodec(), IdentityCodec()])

    def test_unknown_raises(self):
        registry = CompressionRegistry()
        with pytest.raises(UnsupportedCompressionError):
            registry.resolve("unknown")

    def test_with_builtin(self):
        registry = CompressionRegistry.with_builtin()
        assert set(registry.list()) == {"builtin.identity", "builtin.gzip"}


class TestRuntimePipelineCompression:
    def test_pipeline_write_read_identity(self):
        store = MemorySnapshotStore()
        serializer_registry = SerializerRegistry.with_builtin()
        compression_registry = CompressionRegistry.with_builtin()
        pipeline = RuntimePipeline(
            store=store,
            serializer_resolver=serializer_registry,
            compression_resolver=compression_registry,
        )

        snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"test": "data"},
        )
        sid = SnapshotId.new()

        pipeline.write(sid, snapshot, codec_id="builtin.identity")
        restored = pipeline.read(sid)

        assert restored.schema_version == snapshot.schema_version
        assert restored.to_mapping() == snapshot.to_mapping()

    def test_pipeline_write_read_gzip(self):
        store = MemorySnapshotStore()
        serializer_registry = SerializerRegistry.with_builtin()
        compression_registry = CompressionRegistry.with_builtin()
        pipeline = RuntimePipeline(
            store=store,
            serializer_resolver=serializer_registry,
            compression_resolver=compression_registry,
        )

        snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"test": "data" * 100},
        )
        sid = SnapshotId.new()

        pipeline.write(sid, snapshot, codec_id="builtin.gzip")
        restored = pipeline.read(sid)

        assert restored.schema_version == snapshot.schema_version
        assert restored.to_mapping() == snapshot.to_mapping()

        record = store.read(sid)
        assert record.metadata.codec_id == "builtin.gzip"
        assert record.metadata.content_size > record.metadata.stored_size

    def test_pipeline_unknown_compression_raises(self):
        store = MemorySnapshotStore()
        serializer_registry = SerializerRegistry.with_builtin()
        compression_registry = CompressionRegistry()
        pipeline = RuntimePipeline(
            store=store,
            serializer_resolver=serializer_registry,
            compression_resolver=compression_registry,
        )

        snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"test": "data"},
        )
        sid = SnapshotId.new()

        with pytest.raises(UnsupportedCompressionError):
            pipeline.write(sid, snapshot, codec_id="unknown")