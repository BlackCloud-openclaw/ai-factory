# tests/integration/remote/test_s3_integration.py
"""
B4.4: S3 远程存储端到端集成测试（使用 moto 模拟）
"""

import pytest
import boto3
from moto import mock_aws

from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.remote import create_remote_transport, create_default_registry
from src.writing.snapshot.runtime.remote.s3 import S3Config
from src.writing.snapshot.runtime.chunking import Chunk
from src.writing.snapshot.migration import RawSnapshot, SchemaVersion
from src.writing.snapshot.runtime.incremental import VersionNotFoundError


@pytest.fixture
def s3_config():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="test-bucket")
        yield S3Config(
            bucket="test-bucket",
            prefix="test/",
            region="us-east-1",
            access_key="test",
            secret_key="test",
        )


@pytest.fixture
def transport(s3_config):
    serializer_registry, compression_registry = create_default_registry()
    return create_remote_transport(
        s3_config,
        serializer_registry=serializer_registry,
        compression_registry=compression_registry,
        use_cache=True,
        default_serializer_id="builtin.json",
        default_codec_id="builtin.identity",
    )


class TestS3Integration:
    def test_write_and_read_snapshot(self, transport):
        sid = SnapshotId.new()
        snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"name": "test", "value": 42},
        )

        transport.write(sid, snapshot)
        restored = transport.read(sid)

        assert restored.schema_version == snapshot.schema_version
        assert restored.to_mapping() == snapshot.to_mapping()

    def test_write_delta_chain(self, transport):
        base_id = SnapshotId.new()
        base_snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"a": 1},
        )
        transport.write(base_id, base_snapshot)

        delta_id = SnapshotId.new()
        delta_snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"a": 1, "b": 2},
        )
        transport.write(delta_id, delta_snapshot)

        restored = transport.read(delta_id)
        # 直接比较整个 mapping
        assert restored.to_mapping() == delta_snapshot.to_mapping()

    def test_chunks_without_manifest_are_invisible(self, transport):
        sid = SnapshotId.new()
        # 获取底层的 chunk_store（Cached → Remote → chunk_store）
        repo = transport._repository
        # 如果是 CachedChunkRepository，需要穿透到 remote
        if hasattr(repo, "_remote"):
            repo = repo._remote
        chunk_store = repo._chunk_store
        chunk = Chunk(chunk_id=1, payload=b"test_data")

        chunk_store.write_chunk(sid, chunk)

        with pytest.raises(VersionNotFoundError):
            transport.read(sid)

        snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"test": "data"},
        )
        transport.write(sid, snapshot)
        restored = transport.read(sid)
        assert restored.schema_version == snapshot.schema_version

    def test_cache_decorator(self, s3_config):
        serializer_registry, compression_registry = create_default_registry()
        transport_with_cache = create_remote_transport(
            s3_config,
            serializer_registry=serializer_registry,
            compression_registry=compression_registry,
            use_cache=True,
            max_cache_entries=10,
        )
        sid = SnapshotId.new()
        snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"cache_test": "value"},
        )

        transport_with_cache.write(sid, snapshot)
        restored = transport_with_cache.read(sid)
        assert restored.to_mapping() == snapshot.to_mapping()

    def test_supports_optimistic(self, transport):
        repo = transport._repository
        assert hasattr(repo, "supports_optimistic")
        assert repo.supports_optimistic is not None