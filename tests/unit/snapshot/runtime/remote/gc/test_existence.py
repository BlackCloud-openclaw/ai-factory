# tests/unit/snapshot/runtime/remote/gc/test_existence.py

import pytest
from unittest.mock import Mock

from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.remote.gc import (
    ChunkRef,
    S3ChunkExistenceChecker,
    EnumeratorExistenceChecker,
)
from src.writing.snapshot.runtime.remote.s3 import S3Client, S3KeyLayout, S3Config
from src.writing.snapshot.runtime.remote.s3.errors import S3NotFoundError, S3Error


class TestS3ChunkExistenceChecker:

    def setup_method(self):
        self.config = S3Config(bucket="test-bucket", prefix="snapshots/")
        self.key_layout = S3KeyLayout(prefix="snapshots/")
        self.client = Mock(spec=S3Client)
        self.checker = S3ChunkExistenceChecker(self.client, self.key_layout)

    def test_exists_returns_true_when_chunk_exists(self):
        sid = SnapshotId.new()
        ref = ChunkRef(sid, 1)
        self.client.head_object.return_value = {"ContentLength": 1024}
        assert self.checker.exists(ref) is True
        self.client.head_object.assert_called_once_with(self.key_layout.chunk_key(sid, 1))

    def test_exists_returns_false_when_chunk_not_exists(self):
        sid = SnapshotId.new()
        ref = ChunkRef(sid, 1)
        self.client.head_object.side_effect = S3NotFoundError("Not found")
        assert self.checker.exists(ref) is False

    def test_exists_propagates_other_errors(self):
        sid = SnapshotId.new()
        ref = ChunkRef(sid, 1)
        self.client.head_object.side_effect = S3Error("Access denied")
        with pytest.raises(S3Error, match="Access denied"):
            self.checker.exists(ref)


class TestEnumeratorExistenceChecker:

    def test_enumeration_mode(self):
        from src.writing.snapshot.runtime.remote.gc.capability import ChunkEnumerator
        enumerator = Mock(spec=ChunkEnumerator)
        sid1 = SnapshotId.new()
        sid2 = SnapshotId.new()
        enumerator.list_all_chunks.return_value = [ChunkRef(sid1, 1), ChunkRef(sid1, 2)]

        checker = EnumeratorExistenceChecker(enumerator)
        assert checker.exists(ChunkRef(sid1, 1)) is True
        assert checker.exists(ChunkRef(sid2, 1)) is False
        enumerator.list_all_chunks.assert_called_once()