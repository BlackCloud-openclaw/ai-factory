# src/writing/snapshot/runtime/chunk_ref.py
"""
ChunkRef — 值对象，用于标识存储系统中的 Chunk。
"""

from dataclasses import dataclass
from .id import SnapshotId


@dataclass(frozen=True)
class ChunkRef:
    """
    Chunk 引用（snapshot_id + chunk_id）。

    Chunk ID 不是全局唯一的，必须与 SnapshotId 绑定。
    """
    snapshot_id: SnapshotId
    chunk_id: int

    def __str__(self) -> str:
        return f"{self.snapshot_id}/chunks/{self.chunk_id:08d}.bin"