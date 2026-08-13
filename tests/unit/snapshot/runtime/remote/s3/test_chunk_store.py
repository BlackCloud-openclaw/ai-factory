# tests/unit/snapshot/runtime/remote/s3/test_chunk_store.py

import pytest
from unittest.mock import Mock

from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.chunking import Chunk
from src.writing.snapshot.runtime.remote.s3 import S3ChunkStore, S3KeyLayout
from src.writing.snapshot.runtime.remote.s3.client import S3Client


class TestS3ChunkStore:
    def setup_method(self):
        self.sid = SnapshotId.new()
        self.chunk = Chunk(5, b"test data")
        self.client = Mock(spec=S3Client)
        self.layout = S3KeyLayout("prefix/")
        self.store = S3ChunkStore(self.client, self.layout)

    def test_write_chunk(self):
        self.store.write_chunk(self.sid, self.chunk)
        expected_key = f"prefix/{self.sid.value}/chunks/00000005.bin"
        self.client.put_object.assert_called_once_with(expected_key, b"test data")

    def test_read_chunk(self):
        self.client.get_object.return_value = b"read data"
        chunk = self.store.read_chunk(self.sid, 5)
        assert chunk.chunk_id == 5
        assert chunk.payload == b"read data"
        expected_key = f"prefix/{self.sid.value}/chunks/00000005.bin"
        self.client.get_object.assert_called_once_with(expected_key)

    def test_list_chunks(self):
        self.client.list_objects.return_value = [
            f"prefix/{self.sid.value}/chunks/00000001.bin",
            f"prefix/{self.sid.value}/chunks/00000002.bin",
            f"prefix/{self.sid.value}/chunks/00000003.bin",
        ]
        ids = self.store.list_chunks(self.sid)
        assert list(ids) == [1, 2, 3]
        expected_prefix = f"prefix/{self.sid.value}/chunks/"
        self.client.list_objects.assert_called_once_with(expected_prefix)

    def test_delete(self):
        self.client.list_objects.return_value = [f"prefix/{self.sid.value}/chunks/00000001.bin"]
        self.store.delete(self.sid)
        self.client.delete_objects.assert_called_once()