# src/writing/snapshot/runtime/incremental/chunk_repository.py
"""
B3.4/B3.5: ChunkRepository — 存储和加载 ChunkSet / DeltaChunkSet
"""

from typing import Any, Iterator, Mapping, Protocol, Union

from ..id import SnapshotId
from .chunk_set import ChunkSet
from .delta_chunk_set import DeltaChunkSet
from .version_manifest import VersionManifest


class ChunkRepository(Protocol):
    """存储和加载 ChunkSet / DeltaChunkSet 的抽象层。"""

    # ========== B3.4 一次性接口 ==========

    def save_version(
        self,
        snapshot_id: SnapshotId,
        chunks: Union[ChunkSet, DeltaChunkSet],
        parent_id: SnapshotId | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """
        保存版本。
        - 若 parent_id 为 None，存储为 Base（ChunkSet）
        - 若 parent_id 存在，存储为 Delta（DeltaChunkSet）
        """
        ...

    def load_version(self, snapshot_id: SnapshotId) -> Union[ChunkSet, DeltaChunkSet]:
        """加载版本，返回 ChunkSet（Base）或 DeltaChunkSet（Delta）。"""
        ...

    def load_manifest(self, snapshot_id: SnapshotId) -> VersionManifest:
        """加载版本的元数据。"""
        ...

    def exists(self, snapshot_id: SnapshotId) -> bool:
        ...

    def delete(self, snapshot_id: SnapshotId) -> None:
        ...

    def list_ids(self) -> list[SnapshotId]:
        """列出所有已存储的版本 ID。"""
        ...

    # ========== B3.5 流式接口（仅 Base） ==========

    def save_chunk_stream(
        self,
        snapshot_id: SnapshotId,
        chunks: Iterator["Chunk"],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """
        保存完整 Chunk 流（仅用于 Base Snapshot）。
        Delta 必须使用 save_version()。
        """
        ...

    def load_chunk_stream(self, snapshot_id: SnapshotId) -> Iterator["Chunk"]:
        """
        流式加载 Chunk 序列。

        Returns:
            Chunk 迭代器（可用于流式重建）
        """
        ...