import pytest
from types import MappingProxyType

from src.writing.snapshot.migration import (
    RawSnapshot,
    SchemaVersion,
    deep_freeze,
)


class TestDeepFreeze:
    def test_deep_freeze_dict(self):
        data = {"a": {"b": [1, 2]}, "c": {3, 4}}
        frozen = deep_freeze(data)
        assert isinstance(frozen, MappingProxyType)
        assert isinstance(frozen["a"], MappingProxyType)
        assert isinstance(frozen["a"]["b"], tuple)
        assert isinstance(frozen["c"], frozenset)

    def test_deep_freeze_mutability(self):
        data = {"list": [1, 2]}
        frozen = deep_freeze(data)
        data["list"].append(3)
        assert frozen["list"] == (1, 2)


class TestRawSnapshot:
    def test_from_mapping(self):
        data = {"title": "test", "outline": {"chapters": 10}}
        snap = RawSnapshot.from_mapping(
            schema_version=SchemaVersion.parse("1.0"),
            data=data,
        )
        assert snap.schema_version == SchemaVersion(1, 0)
        assert snap.get("title") == "test"
        assert isinstance(snap.get("outline"), MappingProxyType)

    def test_get_default(self):
        snap = RawSnapshot.from_mapping(
            schema_version=SchemaVersion.parse("1.0"),
            data={"exists": True},
        )
        assert snap.get("exists") is True
        assert snap.get("missing") is None
        assert snap.get("missing", "default") == "default"

    def test_require(self):
        snap = RawSnapshot.from_mapping(
            schema_version=SchemaVersion.parse("1.0"),
            data={"exists": 42},
        )
        assert snap.require("exists") == 42
        with pytest.raises(KeyError):
            snap.require("missing")

    def test_fields_cached(self):
        snap = RawSnapshot.from_mapping(
            schema_version=SchemaVersion.parse("1.0"),
            data={"a": 1, "b": 2, "c": 3},
        )
        fields1 = snap.fields()
        fields2 = snap.fields()
        assert fields1 is fields2
        assert fields1 == frozenset({"a", "b", "c"})

    def test_contains(self):
        snap = RawSnapshot.from_mapping(
            schema_version=SchemaVersion.parse("1.0"),
            data={"present": True},
        )
        assert "present" in snap
        assert "missing" not in snap
        assert 123 not in snap

    def test_immutable_nested(self):
        data = {"nested": {"x": [1, 2, 3]}}
        snap = RawSnapshot.from_mapping(
            schema_version=SchemaVersion.parse("1.0"),
            data=data,
        )
        nested = snap.get("nested")
        with pytest.raises(Exception):
            nested["x"] = [4, 5, 6]  # type: ignore

        x = nested.get("x")  # type: ignore
        with pytest.raises(Exception):
            x.append(4)  # type: ignore

    # ========== 新增测试（已修正 self） ==========

    def test_deep_freeze_accepts_mapping_proxy(self):
        data = MappingProxyType({"a": {"b": [1, 2]}})
        frozen = deep_freeze(data)
        assert isinstance(frozen, MappingProxyType)
        assert isinstance(frozen["a"], MappingProxyType)
        assert isinstance(frozen["a"]["b"], tuple)

    def test_from_mapping_accepts_mapping_proxy(self):
        data = MappingProxyType({"nested": {"value": 1}})
        snap = RawSnapshot.from_mapping(
            schema_version=SchemaVersion.parse("1.0"),
            data=data,
        )
        assert isinstance(snap.get("nested"), MappingProxyType)

    def test_original_data_mutation_does_not_affect_snapshot(self):
        data = {"nested": {"value": 1}}
        snap = RawSnapshot.from_mapping(
            schema_version=SchemaVersion.parse("1.0"),
            data=data,
        )
        data["nested"]["value"] = 999
        assert snap["nested"]["value"] == 1

    def test_mapping_protocol(self):
        data = {"title": "test", "chapters": 10}
        snap = RawSnapshot.from_mapping(
            schema_version=SchemaVersion.parse("1.0"),
            data=data,
        )
        assert snap["title"] == "test"
        # 顺序可能与插入顺序一致（Python 3.7+），但仅测试集合包含关系
        keys = list(iter(snap))
        assert "title" in keys
        assert "chapters" in keys
        assert len(snap) == 2