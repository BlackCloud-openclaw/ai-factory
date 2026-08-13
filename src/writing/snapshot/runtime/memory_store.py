# src/writing/snapshot/runtime/memory_store.py
"""
B3.1: MemorySnapshotStore — 内存存储实现（测试用）
"""

from typing import Iterable

from .exceptions import SnapshotNotFoundError
from .id import SnapshotId
from .protocols import SnapshotStore
from .record import SnapshotRecord


class MemorySnapshotStore:
    """内存存储实现，用于测试和轻量级场景。"""

    def __init__(self):
        self._storage: dict[SnapshotId, SnapshotRecord] = {}

    def read(self, snapshot_id: SnapshotId) -> SnapshotRecord:
        if snapshot_id not in self._storage:
            raise SnapshotNotFoundError(f"Snapshot not found: {snapshot_id}")
        return self._storage[snapshot_id]

    def write(self, snapshot_id: SnapshotId, record: SnapshotRecord) -> None:
        self._storage[snapshot_id] = record

    def exists(self, snapshot_id: SnapshotId) -> bool:
        return snapshot_id in self._storage

    def delete(self, snapshot_id: SnapshotId) -> None:
        if snapshot_id in self._storage:
            del self._storage[snapshot_id]

    def list(self) -> Iterable[SnapshotId]:
        return list(self._storage.keys())

    def clear(self) -> None:
        """清空所有存储（仅测试用）。"""
        self._storage.clear()