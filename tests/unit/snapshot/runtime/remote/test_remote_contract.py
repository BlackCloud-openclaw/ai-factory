# tests/unit/snapshot/runtime/remote/test_remote_contract.py

import pytest

from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.chunking import Chunk
from src.writing.snapshot.runtime.incremental import (
    ChunkSet,
    DeltaChunkSet,
    VersionManifest,
    VersionNotFoundError,
)
from src.writing.snapshot.runtime.remote import RemoteChunkRepository


class FakeChunkStore:
    def __init__(self):
        self._chunks: dict[SnapshotId, dict[int, Chunk]] = {}

    def write_chunk(self, snapshot_id: SnapshotId, chunk: Chunk) -> None:
        if snapshot_id not in self._chunks:
            self._chunks[snapshot_id] = {}
        self._chunks[snapshot_id][chunk.chunk_id] = chunk

    def read_chunk(self, snapshot_id: SnapshotId, chunk_id: int) -> Chunk:
        if snapshot_id not in self._chunks or chunk_id not in self._chunks[snapshot_id]:
            raise ValueError("Chunk not found")
        return self._chunks[snapshot_id][chunk_id]

    def list_chunks(self, snapshot_id: SnapshotId) -> list[int]:
        return list(self._chunks.get(snapshot_id, {}).keys())

    def delete(self, snapshot_id: SnapshotId) -> None:
        self._chunks.pop(snapshot_id, None)


class FakeVersionStore:
    def __init__(self):
        self._manifests: dict[SnapshotId, VersionManifest] = {}

    def put(self, manifest: VersionManifest) -> None:
        self._manifests[manifest.snapshot_id] = manifest

    def get(self, snapshot_id: SnapshotId) -> VersionManifest:
        if snapshot_id not in self._manifests:
            raise VersionNotFoundError(f"Version not found: {snapshot_id}")
        return self._manifests[snapshot_id]

    def delete(self, snapshot_id: SnapshotId) -> None:
        self._manifests.pop(snapshot_id, None)

    def list_ids(self) -> list[SnapshotId]:
        return list(self._manifests.keys())


class TestRemoteChunkRepository:
    def setup_method(self):
        self.chunk_store = FakeChunkStore()
        self.version_store = FakeVersionStore()
        self.repo = RemoteChunkRepository(self.chunk_store, self.version_store)

    def test_save_and_load_base(self):
        sid = SnapshotId.new()
        chunks = ChunkSet.from_mapping({1: Chunk(1, b"a"), 2: Chunk(2, b"b")})
        self.repo.save_version(sid, chunks, parent_id=None)

        loaded = self.repo.load_version(sid)
        assert isinstance(loaded, ChunkSet)
        assert loaded == chunks

        manifest = self.repo.load_manifest(sid)
        assert manifest.parent_id is None
        assert manifest.metadata.get("storage_mode") == "base"

    def test_save_and_load_delta(self):
        sid = SnapshotId.new()
        parent = SnapshotId.new()
        delta = DeltaChunkSet(
            added_or_modified={1: Chunk(1, b"new")},
            deleted=frozenset({2}),
        )
        self.repo.save_version(sid, delta, parent_id=parent)

        loaded = self.repo.load_version(sid)
        assert isinstance(loaded, DeltaChunkSet)
        assert loaded == delta

        manifest = self.repo.load_manifest(sid)
        assert manifest.parent_id == parent
        assert manifest.metadata["storage_mode"] == "delta"
        assert manifest.metadata["deleted"] == [2]

    def test_exists(self):
        sid = SnapshotId.new()
        assert self.repo.exists(sid) is False
        self.repo.save_version(sid, ChunkSet.empty())
        assert self.repo.exists(sid) is True

    def test_delete_without_force_raises_if_children(self):
        parent = SnapshotId.new()
        child = SnapshotId.new()
        self.repo.save_version(parent, ChunkSet.empty(), parent_id=None)
        self.repo.save_version(child, ChunkSet.empty(), parent_id=parent)

        from src.writing.snapshot.runtime.remote import SnapshotHasChildrenError
        with pytest.raises(SnapshotHasChildrenError, match="has child version"):
            self.repo.delete(parent, force=False)

    def test_delete_with_force_removes_parent(self):
        parent = SnapshotId.new()
        child = SnapshotId.new()
        self.repo.save_version(parent, ChunkSet.empty(), parent_id=None)
        self.repo.save_version(child, ChunkSet.empty(), parent_id=parent)

        self.repo.delete(parent, force=True)
        assert self.repo.exists(parent) is False

    def test_delete_cascade(self):
        sid = SnapshotId.new()
        self.repo.save_version(sid, ChunkSet.empty(), parent_id=None)
        self.repo.delete(sid)
        assert self.repo.exists(sid) is False

    def test_list_ids(self):
        ids = [SnapshotId.new() for _ in range(3)]
        for sid in ids:
            self.repo.save_version(sid, ChunkSet.empty())
        assert set(self.repo.list_ids()) == set(ids)