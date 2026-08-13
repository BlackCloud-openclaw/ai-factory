# tests/unit/snapshot/runtime/test_record.py

import pytest

from src.writing.snapshot.runtime import SnapshotMetadata, SnapshotRecord


class TestSnapshotRecord:
    def test_record_creation_valid(self):
        metadata = SnapshotMetadata(
            content_size=5,
            stored_size=5,
        )
        record = SnapshotRecord(metadata=metadata, payload=b"hello")
        assert record.metadata == metadata
        assert record.payload == b"hello"

    def test_record_creation_valid_with_stored_size_mismatch(self):
        metadata = SnapshotMetadata(
            content_size=5,
            stored_size=3,
        )
        with pytest.raises(ValueError, match="Payload size 5 does not match"):
            SnapshotRecord(metadata=metadata, payload=b"hello")

    def test_metadata_defaults(self):
        metadata = SnapshotMetadata()
        assert metadata.format_version == 1
        assert metadata.serializer == "builtin.json"   # 修正
        assert metadata.codec_id == "builtin.identity"
        assert metadata.content_size == 0
        assert metadata.stored_size == 0