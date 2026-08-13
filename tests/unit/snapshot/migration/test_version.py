import pytest
from datetime import datetime, timezone

from src.writing.snapshot.migration import (
    CapabilityId,
    FixedClock,
    FixedRandom,
    MigrationContext,
    MigrationEdge,
    SchemaVersion,
    VersionNode,
    VersionType,
    freeze_mapping,
)


class TestSchemaVersion:
    def test_parse_strict_no_normalize(self) -> None:
        v = SchemaVersion.parse("  1.0  ")
        assert v == SchemaVersion(1, 0)

    def test_components_property(self) -> None:
        v = SchemaVersion(1, 2)
        assert v.components == (1, 2)

    def test_next_minor(self) -> None:
        assert SchemaVersion(1, 0).next_minor() == SchemaVersion(1, 1)

    def test_next_major(self) -> None:
        assert SchemaVersion(1, 9).next_major() == SchemaVersion(2, 0)


class TestCapabilityId:
    def test_parse_strict(self) -> None:
        c = CapabilityId.parse("builtin.planning")
        assert c.value == "builtin.planning"

    def test_parse_rejects_uppercase(self) -> None:
        with pytest.raises(ValueError):
            CapabilityId.parse("Builtin.Planning")

    def test_parse_rejects_leading_trailing_whitespace(self) -> None:
        # 注意：parse 是严格模式，不会 strip
        with pytest.raises(ValueError):
            CapabilityId.parse("  builtin.planning  ")


class TestFixedClock:
    def test_utc_aware(self) -> None:
        clock = FixedClock()
        assert clock.now().tzinfo == timezone.utc


class TestFixedRandom:
    def test_generate(self) -> None:
        rng = FixedRandom()
        assert rng.generate() == "fixed-seed"


class TestFreezeMapping:
    def test_freeze(self) -> None:
        d = {"a": 1}
        frozen = freeze_mapping(d)
        assert frozen["a"] == 1
        d["b"] = 2
        assert "b" not in frozen
        with pytest.raises(Exception):
            frozen["x"] = 3  # type: ignore


class TestMigrationEdge:
    def test_valid_edge(self):
        def upcaster(snap, ctx):
            return snap

        edge = MigrationEdge(
            from_version=SchemaVersion(1, 0),
            to_version=SchemaVersion(1, 1),
            upcaster=upcaster,
        )
        assert edge.from_version == SchemaVersion(1, 0)
        assert edge.to_version == SchemaVersion(1, 1)
    

class TestVersionNode:
    def test_metadata_frozen(self) -> None:
        d = {"a": "b"}
        node = VersionNode(
            version=SchemaVersion(1, 0),
            version_type=VersionType.MINOR,
            metadata=d,
        )
        d["c"] = "d"
        assert "c" not in node.metadata