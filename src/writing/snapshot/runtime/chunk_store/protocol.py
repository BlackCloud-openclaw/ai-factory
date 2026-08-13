# src/writing/snapshot/runtime/chunk_store/protocol.py
"""
B3.3: ChunkStore Protocol
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Protocol

from ..id import SnapshotId

if TYPE_CHECKING:
    from ..chunking import Chunk, StreamingManifest


class ChunkReader(Protocol):
    """Chunk 读取协议。"""

    def list_chunks(self) -> Iterable[int]:
        """返回所有 chunk_id（不假定连续）。"""
        ...

    def read_chunk(self, chunk_id: int) -> Chunk:
        """读取指定 chunk_id 的 Chunk。"""
        ...


class ChunkWriter(Protocol):
    """Chunk 写入协议。"""

    def append(self, chunk: Chunk) -> None:
        """追加一个 Chunk。"""
        ...


class ChunkStore(Protocol):
    """分块存储协议。"""

    def create_writer(self, snapshot_id: SnapshotId) -> ChunkWriter:
        """创建写入器。"""
        ...

    def create_reader(self, snapshot_id: SnapshotId) -> ChunkReader:
        """创建读取器。"""
        ...

    def write_manifest(self, snapshot_id: SnapshotId, manifest: StreamingManifest) -> None:
        """写入 Manifest。"""
        ...

    def read_manifest(self, snapshot_id: SnapshotId) -> StreamingManifest:
        """读取 Manifest。"""
        ...

    def delete(self, snapshot_id: SnapshotId) -> None:
        """删除整个快照。"""
        ...