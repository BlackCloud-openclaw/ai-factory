# tests/unit/snapshot/migration/test_schema.py

from src.writing.snapshot.migration import (
    CURRENT_SCHEMA_VERSION,
    MINIMUM_SUPPORTED_VERSION,
    SchemaVersion,
)


class TestSchemaConfig:
    def test_current_version_defined(self):
        assert CURRENT_SCHEMA_VERSION == SchemaVersion(2, 0)

    def test_minimum_supported_defined(self):
        assert MINIMUM_SUPPORTED_VERSION == SchemaVersion(1, 0)

    def test_current_is_newer_than_minimum(self):
        assert CURRENT_SCHEMA_VERSION.is_newer_than(MINIMUM_SUPPORTED_VERSION) is True