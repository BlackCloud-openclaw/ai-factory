# src/writing/snapshot/runtime/serializer.py
"""
B3.1: JsonSerializer — 默认 SnapshotSerializer 实现
"""

import json
from collections.abc import Mapping
from types import MappingProxyType

from .exceptions import SnapshotSerializationError
from .protocols import SnapshotSerializer
from ..migration import RawSnapshot, SchemaVersion

_SCHEMA_VERSION_FIELD = "schema_version"
_DATA_FIELD = "data"


def _to_serializable(obj):
    """递归将 MappingProxyType 转为 dict，并确保嵌套字典键排序。"""
    if isinstance(obj, MappingProxyType):
        return {k: _to_serializable(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


class JsonSerializer:
    name = "json"

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