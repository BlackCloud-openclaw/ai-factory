# tests/unit/snapshot/runtime/test_serializer.py

import json

import pytest

from src.writing.snapshot.runtime import (
    SnapshotSerializationError,
)
from src.writing.snapshot.runtime.serializers import (
    JsonSerializer,
    SnapshotSerializer,  # 从 serializers 导入新版协议
)
from src.writing.snapshot.migration import RawSnapshot, SchemaVersion


class TestJsonSerializer:
    def test_implements_protocol(self):
        """验证 JsonSerializer 符合 SnapshotSerializer Protocol。"""
        assert isinstance(JsonSerializer(), SnapshotSerializer)

    def test_id_is_stable(self):
        serializer = JsonSerializer()
        assert serializer.id == "builtin.json"

    def test_serialize_deterministic(self):
        serializer = JsonSerializer()
        snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"b": 2, "a": 1},
        )
        bytes1 = serializer.serialize(snapshot)
        bytes2 = serializer.serialize(snapshot)
        assert bytes1 == bytes2

    def test_canonical_json_stable(self):
        snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion.from_string("1.0"),
            data={"a": 1, "b": 2},
        )
        payload = JsonSerializer().serialize(snapshot)
        expected = b'{"data":{"a":1,"b":2},"schema_version":"1.0"}'
        assert payload == expected

    def test_round_trip(self):
        serializer = JsonSerializer()
        original = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(2, 0),
            data={"title": "Test", "nested": {"value": 42}},
        )
        payload = serializer.serialize(original)
        restored = serializer.deserialize(payload)
        assert restored.schema_version == original.schema_version
        assert restored.to_mapping() == original.to_mapping()

    def test_deserialize_invalid_json_raises(self):
        serializer = JsonSerializer()
        with pytest.raises(SnapshotSerializationError, match="Invalid JSON payload"):
            serializer.deserialize(b"not json")

    def test_deserialize_missing_schema_version_raises(self):
        serializer = JsonSerializer()
        payload = json.dumps({"data": {}}).encode("utf-8")
        with pytest.raises(SnapshotSerializationError, match="Missing 'schema_version'"):
            serializer.deserialize(payload)

    def test_deserialize_invalid_data_field_raises(self):
        serializer = JsonSerializer()
        payload = json.dumps({
            "schema_version": "1.0",
            "data": "not a dict",
        }).encode("utf-8")
        with pytest.raises(SnapshotSerializationError, match="Missing or invalid 'data'"):
            serializer.deserialize(payload)

    def test_deserialize_invalid_schema_version_raises(self):
        serializer = JsonSerializer()
        payload = json.dumps({
            "schema_version": "invalid",
            "data": {},
        }).encode("utf-8")
        with pytest.raises(SnapshotSerializationError, match="Invalid schema version"):
            serializer.deserialize(payload)

    def test_serializer_is_stateless(self):
        serializer = JsonSerializer()
        snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"x": 1},
        )
        payload = serializer.serialize(snapshot)
        restored = serializer.deserialize(payload)
        assert restored.schema_version == SchemaVersion(1, 0)