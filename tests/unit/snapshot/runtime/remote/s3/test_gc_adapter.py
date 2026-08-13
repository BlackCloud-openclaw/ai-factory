# tests/unit/snapshot/runtime/remote/s3/test_gc_adapter.py

import pytest
from unittest.mock import Mock, call

from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.remote.s3 import S3GCAdapter, S3Client, S3KeyLayout
from src.writing.snapshot.runtime.remote.gc import ChunkRef


class TestS3GCAdapter:
    def setup_method(self):
        self.client = Mock(spec=S3Client)
        self.layout = S3KeyLayout("snapshots/")
        self.adapter = S3GCAdapter(self.client, self.layout)

    def test_list_all_chunks(self):
        sid1 = SnapshotId.new()
        sid2 = SnapshotId.new()
        keys = [
            f"snapshots/{sid1}/chunks/00000001.bin",
            f"snapshots/{sid1}/chunks/00000002.bin",
            f"snapshots/{sid2}/chunks/00000001.bin",
            "snapshots/other/not_chunk.txt",
        ]
        self.client.list_objects.return_value = keys

        chunks = list(self.adapter.list_all_chunks())

        assert len(chunks) == 3
        assert ChunkRef(sid1, 1) in chunks
        assert ChunkRef(sid1, 2) in chunks
        assert ChunkRef(sid2, 1) in chunks
        self.client.list_objects.assert_called_once_with("snapshots/")

    def test_list_chunks(self):
        sid = SnapshotId.new()
        keys = [
            f"snapshots/{sid}/chunks/00000001.bin",
            f"snapshots/{sid}/chunks/00000002.bin",
        ]
        self.client.list_objects.return_value = keys

        chunks = list(self.adapter.list_chunks(sid))

        assert len(chunks) == 2
        assert ChunkRef(sid, 1) in chunks
        assert ChunkRef(sid, 2) in chunks
        expected_prefix = f"snapshots/{sid}/chunks/"
        self.client.list_objects.assert_called_once_with(expected_prefix)

    def test_parse_chunk_key_via_layout(self):
        sid = SnapshotId.new()
        key = f"snapshots/{sid}/chunks/00000042.bin"
        ref = self.layout.parse_chunk_key(key)
        assert ref == ChunkRef(sid, 42)

    def test_parse_chunk_key_invalid(self):
        assert self.layout.parse_chunk_key("snapshots/not_a_snapshot/file.txt") is None
        assert self.layout.parse_chunk_key("snapshots/invalid/chunks/not_number.bin") is None

    def test_delete_chunks_batches(self):
        sid1 = SnapshotId.new()
        sid2 = SnapshotId.new()
        # 创建 1500 个 chunk 引用
        chunks = []
        for i in range(1500):
            chunks.append(ChunkRef(sid1, i))

        self.adapter.delete_chunks(chunks)

        # 验证调用了 2 次 delete_objects（1000 + 500）
        assert self.client.delete_objects.call_count == 2
        # 第一次调用 1000 个
        first_call_keys = self.client.delete_objects.call_args_list[0][0][0]
        assert len(first_call_keys) == 1000
        # 第二次调用 500 个
        second_call_keys = self.client.delete_objects.call_args_list[1][0][0]
        assert len(second_call_keys) == 500

    def test_delete_chunks_empty(self):
        self.adapter.delete_chunks([])
        self.client.delete_objects.assert_not_called()