# src/writing/snapshot/runtime/protocols.py
"""
B3: Runtime Protocol 定义
"""

from typing import Iterable, Protocol, runtime_checkable

from .id import SnapshotId
from .record import SnapshotRecord
from src.writing.snapshot.migration import RawSnapshot  # 绝对导入


@runtime_checkable
class SnapshotSerializer(Protocol):
    @property
    def name(self) -> str:
        ...

    def serialize(self, snapshot: RawSnapshot) -> bytes:
        ...

    def deserialize(self, payload: bytes) -> RawSnapshot:
        ...


@runtime_checkable
class SnapshotStore(Protocol):
    def read(self, snapshot_id: SnapshotId) -> SnapshotRecord:
        ...

    def write(self, snapshot_id: SnapshotId, record: SnapshotRecord) -> None:
        ...

    def exists(self, snapshot_id: SnapshotId) -> bool:
        ...

    def delete(self, snapshot_id: SnapshotId) -> None:
        ...

    def list(self) -> Iterable[SnapshotId]:
        ...