# src/writing/snapshot/runtime/serializers/errors.py
"""
B3.2: Serializer 错误类型
"""

from ..exceptions import SnapshotRuntimeError


class SerializerError(SnapshotRuntimeError):
    pass


class UnsupportedSerializerError(SerializerError):
    pass


class DuplicateSerializerError(SerializerError):
    pass