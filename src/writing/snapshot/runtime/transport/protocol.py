# src/writing/snapshot/runtime/transport/protocol.py
"""
B3.3: Transport Protocol — 运行时传输层
"""

from typing import Protocol

from ..id import SnapshotId
from ...migration import RawSnapshot


class Transport(Protocol):
    """传输层协议，Pipeline 的唯一依赖。"""

    def write(self, snapshot_id: SnapshotId, snapshot: RawSnapshot) -> None:
        """写入快照。"""
        ...

    def read(self, snapshot_id: SnapshotId) -> RawSnapshot:
        """读取快照。"""
        ...