# src/writing/snapshot/runtime/serializers/protocol.py
"""
B3.2: SnapshotSerializer Protocol
"""

from typing import Protocol, runtime_checkable

from src.writing.snapshot.migration import RawSnapshot  # 绝对导入


@runtime_checkable
class SnapshotSerializer(Protocol):
    @property
    def id(self) -> str:
        ...

    @property
    def display_name(self) -> str:
        ...

    def serialize(self, snapshot: RawSnapshot) -> bytes:
        ...

    def deserialize(self, payload: bytes) -> RawSnapshot:
        ...