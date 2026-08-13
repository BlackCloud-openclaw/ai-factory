# tests/unit/snapshot/runtime/test_store.py

import tempfile
from pathlib import Path

import pytest

from src.writing.snapshot.runtime import (
    FileSnapshotStore,
    MemorySnapshotStore,
    SnapshotId,
    SnapshotMetadata,
    SnapshotNotFoundError,
    SnapshotRecord,
)


class TestMemorySnapshotStore:
    def test_write_and_read(self):
        store = MemorySnapshotStore()
        sid = SnapshotId.new()
        metadata = SnapshotMetadata(content_size=5, stored_size=5)
        record = SnapshotRecord(metadata=metadata, payload=b"hello")

        store.write(sid, record)
        assert store.exists(sid) is True

        retrieved = store.read(sid)
        assert retrieved.metadata == metadata
        assert retrieved.payload == b"hello"

    def test_read_not_found(self):
        store = MemorySnapshotStore()
        sid = SnapshotId.new()
        with pytest.raises(SnapshotNotFoundError):
            store.read(sid)

    def test_delete(self):
        store = MemorySnapshotStore()
        sid = SnapshotId.new()
        record = SnapshotRecord(
            metadata=SnapshotMetadata(content_size=5, stored_size=5),
            payload=b"hello",
        )
        store.write(sid, record)
        assert store.exists(sid) is True
        store.delete(sid)
        assert store.exists(sid) is False

    def test_list(self):
        store = MemorySnapshotStore()
        sid1 = SnapshotId.new()
        sid2 = SnapshotId.new()
        record = SnapshotRecord(
            metadata=SnapshotMetadata(content_size=5, stored_size=5),
            payload=b"hello",
        )
        store.write(sid1, record)
        store.write(sid2, record)

        ids = list(store.list())
        assert len(ids) == 2
        assert sid1 in ids
        assert sid2 in ids

    def test_clear(self):
        store = MemorySnapshotStore()
        sid = SnapshotId.new()
        record = SnapshotRecord(
            metadata=SnapshotMetadata(content_size=5, stored_size=5),
            payload=b"hello",
        )
        store.write(sid, record)
        store.clear()
        assert store.exists(sid) is False


class TestFileSnapshotStore:
    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSnapshotStore(Path(tmpdir))
            sid = SnapshotId.new()
            metadata = SnapshotMetadata(content_size=5, stored_size=5)
            record = SnapshotRecord(metadata=metadata, payload=b"hello")

            store.write(sid, record)
            assert store.exists(sid) is True

            retrieved = store.read(sid)
            assert retrieved.metadata == metadata
            assert retrieved.payload == b"hello"

    def test_read_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSnapshotStore(Path(tmpdir))
            sid = SnapshotId.new()
            with pytest.raises(SnapshotNotFoundError):
                store.read(sid)

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSnapshotStore(Path(tmpdir))
            sid = SnapshotId.new()
            record = SnapshotRecord(
                metadata=SnapshotMetadata(content_size=5, stored_size=5),
                payload=b"hello",
            )
            store.write(sid, record)
            assert store.exists(sid) is True
            store.delete(sid)
            assert store.exists(sid) is False

    def test_write_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSnapshotStore(Path(tmpdir))
            sid = SnapshotId.new()

            record1 = SnapshotRecord(
                metadata=SnapshotMetadata(content_size=3, stored_size=3),
                payload=b"foo",
            )
            store.write(sid, record1)
            assert store.read(sid).payload == b"foo"

            record2 = SnapshotRecord(
                metadata=SnapshotMetadata(content_size=3, stored_size=3),
                payload=b"bar",
            )
            store.write(sid, record2)
            assert store.read(sid).payload == b"bar"

    def test_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSnapshotStore(Path(tmpdir))
            sid1 = SnapshotId.new()
            sid2 = SnapshotId.new()
            record = SnapshotRecord(
                metadata=SnapshotMetadata(content_size=5, stored_size=5),
                payload=b"hello",
            )
            store.write(sid1, record)
            store.write(sid2, record)

            ids = list(store.list())
            assert len(ids) == 2
            assert sid1 in ids
            assert sid2 in ids

    def test_list_ignores_non_snapshot_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSnapshotStore(Path(tmpdir))
            # 创建非 Snapshot 文件
            (Path(tmpdir) / "temp.txt").write_text("ignore me")
            (Path(tmpdir) / "log.log").write_text("ignore me")

            sid = SnapshotId.new()
            record = SnapshotRecord(
                metadata=SnapshotMetadata(content_size=5, stored_size=5),
                payload=b"hello",
            )
            store.write(sid, record)

            ids = list(store.list())
            assert len(ids) == 1
            assert sid in ids