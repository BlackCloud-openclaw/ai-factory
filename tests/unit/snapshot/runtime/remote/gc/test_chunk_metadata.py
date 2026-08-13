# tests/unit/snapshot/runtime/remote/gc/test_chunk_metadata.py

import pytest
from datetime import datetime
from unittest.mock import Mock

from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.remote.gc import ChunkRef, ChunkMetadata, ChunkMetadataProvider
from src.writing.snapshot.runtime.remote.s3 import S3GCAdapter, S3Client, S3KeyLayout
from src.writing.snapshot.runtime.remote.s3.errors import S3NotFoundError


class TestChunkMetadata:
    def test_metadata_model(self):
        sid = SnapshotId.new()
        ref = ChunkRef(sid, 123)
        now = datetime.now()
        meta = ChunkMetadata(
            chunk_ref=ref,
            size_bytes=1024,
            checksum="etag123",
            created_at=now,
        )
        assert meta.chunk_ref == ref
        assert meta.size_bytes == 1024
        assert meta.checksum == "etag123"
        assert meta.created_at == now

    def test_deletion_candidate_uses_metadata(self):
        sid = SnapshotId.new()
        ref = ChunkRef(sid, 456)
        meta = ChunkMetadata(ref, size_bytes=2048)
        from src.writing.snapshot.runtime.remote.gc import DeletionCandidate
        candidate = DeletionCandidate(metadata=meta)
        assert candidate.chunk_ref == ref
        assert candidate.size_bytes == 2048


class TestS3MetadataProvider:
    def setup_method(self):
        self.client = Mock(spec=S3Client)
        self.layout = S3KeyLayout("snapshots/")
        self.adapter = S3GCAdapter(self.client, self.layout)

    def test_get_metadata(self):
        sid = SnapshotId.new()
        ref = ChunkRef(sid, 1)
        key = f"snapshots/{sid}/chunks/00000001.bin"

        # Mock head_object 响应
        self.client.head_object.return_value = {
            "ContentLength": 1024,
            "ETag": '"abc123"',
            "LastModified": datetime(2026, 1, 1, 12, 0, 0),
        }

        metadata = self.adapter.get_metadata(ref)
        assert metadata.chunk_ref == ref
        assert metadata.size_bytes == 1024
        assert metadata.checksum == '"abc123"'
        assert metadata.created_at == datetime(2026, 1, 1, 12, 0, 0)

        self.client.head_object.assert_called_once_with(key)

    def test_get_metadata_not_found(self):
        sid = SnapshotId.new()
        ref = ChunkRef(sid, 99)
        self.client.head_object.side_effect = S3NotFoundError("Not found")

        with pytest.raises(ValueError, match="Chunk not found"):
            self.adapter.get_metadata(ref)

    def test_get_metadata_handles_missing_fields(self):
        sid = SnapshotId.new()
        ref = ChunkRef(sid, 1)
        self.client.head_object.return_value = {
            "ContentLength": 0,  # 可能为 0
            # 没有 ETag 和 LastModified
        }

        metadata = self.adapter.get_metadata(ref)
        assert metadata.size_bytes == 0
        assert metadata.checksum is None
        assert metadata.created_at is None

    def test_metadata_provider_is_runtime_checkable(self):
        assert isinstance(self.adapter, ChunkMetadataProvider)