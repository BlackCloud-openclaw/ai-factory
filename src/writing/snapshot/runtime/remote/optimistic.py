# src/writing/snapshot/runtime/remote/optimistic.py
"""
B4: OptimisticChunkRepository Protocol
"""

from typing import Protocol, Union, Mapping, Any

from ..id import SnapshotId
from ..incremental import ChunkSet, DeltaChunkSet


class OptimisticChunkRepository(Protocol):
    """
    乐观锁扩展协议。
    实现此协议的 Repository 支持带预期父版本的写入。
    """

    def save_version(
        self,
        snapshot_id: SnapshotId,
        chunks: Union[ChunkSet, DeltaChunkSet],
        parent_id: SnapshotId | None = None,
        metadata: Mapping[str, Any] | None = None,
        *,
        expected_parent: SnapshotId | None = None,
    ) -> None:
        ...