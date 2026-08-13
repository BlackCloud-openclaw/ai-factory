# src/writing/snapshot/runtime/remote/protocols.py
"""
B4: Remote 存储层 Protocol（仅用于 Remote 实现内部）
"""

from typing import Protocol

from ..id import SnapshotId
from ..incremental import Chunk, DeltaChunkSet, VersionManifest, ChunkSet


class RemoteChunkStore(Protocol):
    """远程 Chunk 存储协议（与本地 ChunkStore 对齐）。"""

    def save_chunks(
        self,
        snapshot_id: SnapshotId,
        chunks: list[Chunk],
    ) -> None:
        """保存一组 Chunk。"""
        ...

    def load_chunks(
        self,
        snapshot_id: SnapshotId,
    ) -> list[Chunk]:
        """加载所有 Chunk。"""
        ...

    def delete_chunks(self, snapshot_id: SnapshotId) -> None:
        ...


class RemoteVersionStore(Protocol):
    """远程版本元数据存储协议。"""

    def save_manifest(self, manifest: VersionManifest) -> None:
        """保存 VersionManifest，原子写入。"""
        ...

    def load_manifest(self, snapshot_id: SnapshotId) -> VersionManifest:
        ...

    def delete_manifest(self, snapshot_id: SnapshotId) -> None:
        ...

    def list_ids(self) -> list[SnapshotId]:
        ...