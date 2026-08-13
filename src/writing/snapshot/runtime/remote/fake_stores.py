# tests/unit/snapshot/runtime/remote/fake_stores.py
"""
测试用假存储实现
"""

from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.chunking import Chunk
from src.writing.snapshot.runtime.incremental import (
    VersionManifest,
    VersionNotFoundError,
)


class FakeChunkStore:
    """测试用假 ChunkStore，不继承 Protocol。"""
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