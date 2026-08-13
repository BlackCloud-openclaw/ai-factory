# src/writing/snapshot/runtime/serializers/registry.py
"""
B3.2: SerializerRegistry — 不可变注册表
"""

from .errors import UnsupportedSerializerError, DuplicateSerializerError
from .protocol import SnapshotSerializer
from .resolver import SerializerResolver


class SerializerRegistry(SerializerResolver):
    """序列化器注册表（不可变，构造时注入）。"""

    def __init__(self, serializers: list[SnapshotSerializer] | None = None):
        self._serializers: dict[str, SnapshotSerializer] = {}
        if serializers:
            for s in serializers:
                if s.id in self._serializers:
                    raise DuplicateSerializerError(
                        f"Serializer with id '{s.id}' already registered"
                    )
                self._serializers[s.id] = s

    def resolve(self, serializer_id: str) -> SnapshotSerializer:
        if serializer_id not in self._serializers:
            raise UnsupportedSerializerError(f"Unknown serializer: {serializer_id}")
        return self._serializers[serializer_id]

    def list(self) -> list[str]:
        return list(self._serializers.keys())

    @classmethod
    def with_builtin(cls) -> "SerializerRegistry":
        from .json_serializer import JsonSerializer
        return cls([JsonSerializer()])