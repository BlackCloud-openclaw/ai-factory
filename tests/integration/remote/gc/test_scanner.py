# tests/integration/remote/gc/test_scanner.py

import pytest
from unittest.mock import Mock

from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.remote.s3.client import S3ObjectSummary
from src.writing.snapshot.runtime.remote.s3 import S3Client, S3KeyLayout, S3Config
from src.writing.snapshot.runtime.remote.s3.errors import S3Error
from src.writing.snapshot.runtime.remote.gc import ChunkRef, S3DeletionMarkerScanner, MarkerScannerError


class TestS3DeletionMarkerScanner:

    def setup_method(self):
        self.config = S3Config(bucket="test-bucket", prefix="snapshots/")
        self.key_layout = S3KeyLayout(prefix="snapshots/")
        self.client = Mock(spec=S3Client)
        self.scanner = S3DeletionMarkerScanner(self.client, self.key_layout)

    def test_iter_pending_markers_empty(self):
        self.client.iter_objects.return_value = iter([])
        result = list(self.scanner.iter_pending_markers())
        assert result == []

    def test_iter_pending_markers_with_valid_markers(self):
        sid1 = SnapshotId.new()
        sid2 = SnapshotId.new()
        summaries = [
            S3ObjectSummary(key=f"snapshots/gc-markers/{sid1.value}/00000001.json", size=100, etag="etag1"),
            S3ObjectSummary(key=f"snapshots/gc-markers/{sid1.value}/00000002.json", size=100, etag="etag2"),
            S3ObjectSummary(key=f"snapshots/gc-markers/{sid2.value}/00000001.json", size=100, etag="etag3"),
            S3ObjectSummary(key="snapshots/other/key.txt", size=100, etag="etag4"),
        ]
        self.client.iter_objects.return_value = iter(summaries)
        result = list(self.scanner.iter_pending_markers())
        assert len(result) == 3
        expected = {ChunkRef(sid1, 1), ChunkRef(sid1, 2), ChunkRef(sid2, 1)}
        assert set(result) == expected

    def test_iter_pending_markers_skips_malformed_keys(self):
        summaries = [
            S3ObjectSummary(key="snapshots/gc-markers/invalid/00000001.json", size=100, etag="etag1"),
            S3ObjectSummary(key="snapshots/gc-markers/abc/not_a_number.json", size=100, etag="etag2"),
        ]
        self.client.iter_objects.return_value = iter(summaries)
        result = list(self.scanner.iter_pending_markers())
        assert result == []

    def test_iter_pending_markers_raises_on_client_error(self):
        self.client.iter_objects.side_effect = S3Error("S3 error")
        with pytest.raises(MarkerScannerError):
            list(self.scanner.iter_pending_markers())

    def test_iter_pending_markers_with_prefix(self):
        sid = SnapshotId.new()
        # 调用时传入的 prefix 必须包含完整前缀
        prefix = f"snapshots/gc-markers/{sid.value}/"
        summaries = [
            S3ObjectSummary(key=f"snapshots/gc-markers/{sid.value}/00000001.json", size=100, etag="etag1"),
        ]
        self.client.iter_objects.return_value = iter(summaries)
        result = list(self.scanner.iter_pending_markers(prefix=prefix))
        assert len(result) == 1
        assert result[0] == ChunkRef(sid, 1)
        self.client.iter_objects.assert_called_with(prefix)

    def test_iter_pending_markers_is_truly_lazy(self):
        called = []
        sid = SnapshotId.new()

        def fake_iter_objects(prefix):
            called.append(1)
            yield S3ObjectSummary(key=f"snapshots/gc-markers/{sid.value}/00000001.json", size=100, etag="etag1")
            called.append(2)
            yield S3ObjectSummary(key=f"snapshots/gc-markers/{sid.value}/00000002.json", size=100, etag="etag2")
            called.append(3)

        self.client.iter_objects = fake_iter_objects

        it = self.scanner.iter_pending_markers()
        first = next(it)
        assert called == [1]
        assert isinstance(first, ChunkRef)
        assert first.chunk_id == 1

        second = next(it)
        assert called == [1, 2]
        assert isinstance(second, ChunkRef)
        assert second.chunk_id == 2