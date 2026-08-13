# src/writing/snapshot/runtime/chunk_store/memory.py
"""
B3.3: MemoryChunkStore — 内存分块存储（测试用）
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from ..id import SnapshotId
from ..exceptions import SnapshotNotFoundError
from .protocol import ChunkReader, ChunkWriter, ChunkStore

if TYPE_CHECKING:
    from ..chunking import Chunk, StreamingManifest


class _MemoryChunkReader:
    def __init__(self, chunks: dict[int, Chunk]):
        self._chunks = chunks

    def list_chunks(self) -> Iterable[int]:
        return sorted(self._chunks.keys())

    def read_chunk(self, chunk_id: int) -> Chunk:
        if chunk_id not in self._chunks:
            raise ValueError(f"Chunk {chunk_id} not found")
        return self._chunks[chunk_id]


class _MemoryChunkWriter:
    def __init__(self, chunks: dict[int, Chunk]):
        self._chunks = chunks

    def append(self, chunk: Chunk) -> None:
        self._chunks[chunk.chunk_id] = chunk


class MemoryChunkStore:
    def __init__(self):
        self._manifests: dict[SnapshotId, StreamingManifest] = {}
        self._chunks: dict[SnapshotId, dict[int, Chunk]] = {}

    def create_writer(self, snapshot_id: SnapshotId) -> ChunkWriter:
        chunks: dict[int, Chunk] = {}
        self._chunks[snapshot_id] = chunks
        return _MemoryChunkWriter(chunks)

    def create_reader(self, snapshot_id: SnapshotId) -> ChunkReader:
        if snapshot_id not in self._chunks:
            raise SnapshotNotFoundError(f"Snapshot not found: {snapshot_id}")
        return _MemoryChunkReader(self._chunks[snapshot_id])

    def write_manifest(self, snapshot_id: SnapshotId, manifest: StreamingManifest) -> None:
        self._manifests[snapshot_id] = manifest

    def read_manifest(self, snapshot_id: SnapshotId) -> StreamingManifest:
        if snapshot_id not in self._manifests:
            raise SnapshotNotFoundError(f"Manifest not found: {snapshot_id}")
        return self._manifests[snapshot_id]

    def delete(self, snapshot_id: SnapshotId) -> None:
        self._manifests.pop(snapshot_id, None)
        self._chunks.pop(snapshot_id, None)