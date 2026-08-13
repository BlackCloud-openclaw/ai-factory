# tests/unit/snapshot/runtime/remote/test_cached_repository.py

import pytest

from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.chunking import Chunk
from src.writing.snapshot.runtime.incremental import (
    ChunkSet,
    VersionManifest,
    VersionNotFoundError,
)
from src.writing.snapshot.runtime.remote import CachedChunkRepository, RemoteChunkRepository


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


class TestCachedChunkRepository:
    def setup_method(self):
        self.remote_store = RemoteChunkRepository(FakeChunkStore(), FakeVersionStore())
        self.repo = CachedChunkRepository(self.remote_store, max_entries=2)

    def test_read_through_manifest(self):
        sid = SnapshotId.new()
        chunks = ChunkSet.from_mapping({1: Chunk(1, b"data")})
        self.remote_store.save_version(sid, chunks, parent_id=None)

        manifest = self.repo.load_manifest(sid)
        assert manifest.parent_id is None
        assert self.repo._manifest_cache.contains(sid) is True

    def test_read_through_version(self):
        sid = SnapshotId.new()
        chunks = ChunkSet.from_mapping({1: Chunk(1, b"data")})
        self.remote_store.save_version(sid, chunks, parent_id=None)

        loaded = self.repo.load_version(sid)
        assert loaded == chunks
        assert self.repo._version_cache.contains(sid) is True

    def test_write_invalidate(self):
        sid = SnapshotId.new()
        chunks = ChunkSet.from_mapping({1: Chunk(1, b"data")})

        # save_version 后缓存应失效
        self.repo.save_version(sid, chunks, parent_id=None)
        assert self.repo._manifest_cache.contains(sid) is False
        assert self.repo._version_cache.contains(sid) is False

        # load_version 后缓存应填充
        loaded = self.repo.load_version(sid)
        assert loaded == chunks
        assert self.repo._version_cache.contains(sid) is True
        assert self.repo._manifest_cache.contains(sid) is True

        # 再次 save_version 应使缓存失效
        chunks2 = ChunkSet.from_mapping({1: Chunk(1, b"new_data")})
        self.repo.save_version(sid, chunks2, parent_id=None)
        assert self.repo._manifest_cache.contains(sid) is False
        assert self.repo._version_cache.contains(sid) is False

        # load_version 应填充新数据
        loaded = self.repo.load_version(sid)
        assert loaded == chunks2
        assert self.repo._version_cache.contains(sid) is True
        assert self.repo._manifest_cache.contains(sid) is True

    def test_eviction(self):
        for i in range(3):
            sid = SnapshotId.new()
            chunks = ChunkSet.from_mapping({1: Chunk(1, f"data{i}".encode())})
            self.repo.save_version(sid, chunks, parent_id=None)
            self.repo.load_version(sid)

        assert self.repo._manifest_cache.size() <= 2
        assert self.repo._version_cache.size() <= 2

    def test_delete_evicts_cache(self):
        sid = SnapshotId.new()
        chunks = ChunkSet.from_mapping({1: Chunk(1, b"data")})
        self.repo.save_version(sid, chunks, parent_id=None)
        self.repo.load_version(sid)
        assert self.repo._version_cache.contains(sid) is True

        self.repo.delete(sid)
        assert self.repo._manifest_cache.contains(sid) is False
        assert self.repo._version_cache.contains(sid) is False

    def test_list_ids_union(self):
        remote_sid = SnapshotId.new()
        self.remote_store.save_version(remote_sid, ChunkSet.empty(), parent_id=None)
        self.repo.load_manifest(remote_sid)

        ids = self.repo.list_ids()
        assert remote_sid in ids

    def test_streaming_bypasses_cache(self):
        sid = SnapshotId.new()
        chunks = [Chunk(1, b"chunk1"), Chunk(2, b"chunk2")]

        self.repo.save_chunk_stream(sid, iter(chunks), metadata={"test": "stream"})
        assert self.repo._manifest_cache.contains(sid) is False

        loaded = list(self.repo.load_chunk_stream(sid))
        assert len(loaded) == 2
        assert loaded[0].payload == b"chunk1"

    def test_metrics(self):
        sid = SnapshotId.new()
        chunks = ChunkSet.from_mapping({1: Chunk(1, b"data")})
        self.remote_store.save_version(sid, chunks, parent_id=None)

        self.repo.load_version(sid)
        metrics = self.repo.metrics()
        assert metrics["version"].hits == 0
        assert metrics["version"].misses == 1

        self.repo.load_version(sid)
        metrics = self.repo.metrics()
        assert metrics["version"].hits == 1
        assert metrics["version"].misses == 1
        assert metrics["version"].hit_rate == 0.5