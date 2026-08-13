# src/writing/snapshot/runtime/serializers/resolver.py
"""
B3.2: SerializerResolver Protocol
"""

from typing import Protocol

from .protocol import SnapshotSerializer


class SerializerResolver(Protocol):
    """根据 serializer_id 获取 SnapshotSerializer。"""

    def resolve(self, serializer_id: str) -> SnapshotSerializer:
        ...