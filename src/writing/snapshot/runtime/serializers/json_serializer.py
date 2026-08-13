# src/writing/snapshot/runtime/serializers/json_serializer.py
"""
B3.2/B3.5: JsonSerializer — 默认 JSON 序列化器（支持流式输出）
"""

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Iterator

from .protocol import SnapshotSerializer
from ..exceptions import SnapshotSerializationError
from ...migration import RawSnapshot, SchemaVersion

_SCHEMA_VERSION_FIELD = "schema_version"
_DATA_FIELD = "data"


def _to_serializable(obj):
    if isinstance(obj, MappingProxyType):
        return {k: _to_serializable(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


class JsonSerializer:
    """基于 JSON 的 RawSnapshot 序列化器（默认实现，支持流式）。"""

    id = "builtin.json"
    display_name = "JSON"

    # ========== B3.2 一次性接口 ==========

    def serialize(self, snapshot: RawSnapshot) -> bytes:
        data = {
            _SCHEMA_VERSION_FIELD: str(snapshot.schema_version),
            _DATA_FIELD: _to_serializable(snapshot.to_mapping()),
        }
        json_str = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return json_str.encode("utf-8")

    def deserialize(self, payload: bytes) -> RawSnapshot:
        try:
            data = json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise SnapshotSerializationError(f"Invalid JSON payload: {e}") from e

        schema_version_str = data.get(_SCHEMA_VERSION_FIELD)
        if not schema_version_str:
            raise SnapshotSerializationError("Missing 'schema_version' in payload")

        raw_data = data.get(_DATA_FIELD)
        if not isinstance(raw_data, Mapping):
            raise SnapshotSerializationError("Missing or invalid 'data' field")

        try:
            schema_version = SchemaVersion.from_string(schema_version_str)
        except ValueError as e:
            raise SnapshotSerializationError(f"Invalid schema version: {schema_version_str}") from e

        return RawSnapshot.from_mapping(
            schema_version=schema_version,
            data=raw_data,
        )

    # ========== B3.5 流式接口 ==========

    def serialize_stream(self, snapshot: RawSnapshot) -> Iterator[bytes]:
        """
        流式序列化，使用 JSONEncoder.iterencode() 逐步输出。

        Yields:
            8KB-64KB 的字节块
        """
        data = {
            _SCHEMA_VERSION_FIELD: str(snapshot.schema_version),
            _DATA_FIELD: _to_serializable(snapshot.to_mapping()),
        }

        encoder = json.JSONEncoder(
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

        chunk_size = 8192
        buffer: list[bytes] = []
        buffer_size = 0

        for token in encoder.iterencode(data):
            encoded = token.encode("utf-8")
            buffer.append(encoded)
            buffer_size += len(encoded)
            if buffer_size >= chunk_size:
                yield b"".join(buffer)
                buffer = []
                buffer_size = 0

        if buffer:
            yield b"".join(buffer)

    def deserialize_stream(self, stream: Iterator[bytes]) -> RawSnapshot:
        """
        从分块字节输入重建 RawSnapshot。

        注意：此方法使用累积缓冲区（bounded buffer），
        在极大型 JSON 输入下可能仍会占用较多内存。
        如需真正的零拷贝流式反序列化，建议使用 ijson 等库，
        但不属于 B3.5 范围。
        """
        buffer = b""
        decoder = json.JSONDecoder()

        for chunk in stream:
            buffer += chunk
            try:
                data, pos = decoder.raw_decode(buffer.decode("utf-8"))
                if pos == len(buffer):
                    return self._deserialize_data(data)
            except json.JSONDecodeError:
                continue

        raise SnapshotSerializationError("Incomplete or invalid JSON stream")

    def _deserialize_data(self, data: dict) -> RawSnapshot:
        """从已解析的 JSON dict 重建 RawSnapshot（内部方法）。"""
        schema_version_str = data.get(_SCHEMA_VERSION_FIELD)
        if not schema_version_str:
            raise SnapshotSerializationError(
                f"Missing '{_SCHEMA_VERSION_FIELD}' in payload"
            )

        raw_data = data.get(_DATA_FIELD)
        if not isinstance(raw_data, Mapping):
            raise SnapshotSerializationError(
                f"Missing or invalid '{_DATA_FIELD}' in payload"
            )

        try:
            schema_version = SchemaVersion.from_string(schema_version_str)
        except ValueError as e:
            raise SnapshotSerializationError(
                f"Invalid schema version: {schema_version_str}"
            ) from e

        return RawSnapshot.from_mapping(
            schema_version=schema_version,
            data=raw_data,
        )