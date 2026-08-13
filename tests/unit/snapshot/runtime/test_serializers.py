# tests/unit/snapshot/runtime/test_serializers.py

import pytest

from src.writing.snapshot.runtime.serializers import (
    SerializerRegistry,
    JsonSerializer,
    UnsupportedSerializerError,
    DuplicateSerializerError,
)
from src.writing.snapshot.migration import RawSnapshot, SchemaVersion


class TestJsonSerializer:
    def test_round_trip(self):
        serializer = JsonSerializer()
        original = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"a": 1, "b": 2},
        )
        payload = serializer.serialize(original)
        restored = serializer.deserialize(payload)
        assert restored.schema_version == original.schema_version
        assert restored.to_mapping() == original.to_mapping()

    def test_canonical_json(self):
        serializer = JsonSerializer()
        snapshot = RawSnapshot.from_mapping(
            schema_version=SchemaVersion(1, 0),
            data={"a": 1, "b": 2},
        )
        payload = serializer.serialize(snapshot)
        expected = b'{"data":{"a":1,"b":2},"schema_version":"1.0"}'
        assert payload == expected

    def test_id(self):
        assert JsonSerializer().id == "builtin.json"


class TestSerializerRegistry:
    def test_register_and_resolve(self):
        registry = SerializerRegistry([JsonSerializer()])
        assert registry.resolve("builtin.json").id == "builtin.json"

    def test_duplicate_raises(self):
        with pytest.raises(DuplicateSerializerError):
            SerializerRegistry([JsonSerializer(), JsonSerializer()])

    def test_unknown_raises(self):
        registry = SerializerRegistry()
        with pytest.raises(UnsupportedSerializerError):
            registry.resolve("unknown")

    def test_with_builtin(self):
        registry = SerializerRegistry.with_builtin()
        assert set(registry.list()) == {"builtin.json"}