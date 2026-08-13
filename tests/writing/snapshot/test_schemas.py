# tests/writing/snapshot/test_schemas.py

import pytest
from uuid import UUID

from src.writing.snapshot.models import (
    PipelineSnapshot,
    SnapshotIdentity,
    SnapshotManifest,
    SnapshotMetadata,
)
from src.writing.snapshot.validation import ValidationResult, Severity, ValidationIssue


class TestSchemas:
    def test_snapshot_identity_has_uuid(self):
        identity = SnapshotIdentity()
        assert isinstance(identity.snapshot_id, UUID)
        assert str(identity.snapshot_id)

    def test_snapshot_manifest_has_defaults(self):
        manifest = SnapshotManifest()
        assert manifest.schema_version == "1.0"
        assert manifest.format_version == "1.0"
        assert manifest.serializer == "json"

    def test_validation_result_separates_severities(self):
        issues = [
            ValidationIssue(Severity.INFO, "INFO_001", "field", "info"),
            ValidationIssue(Severity.WARNING, "WARN_001", "field", "warning"),
            ValidationIssue(Severity.ERROR, "ERR_001", "field", "error"),
        ]
        result = ValidationResult(is_valid=False, issues=issues)
        
        assert len(result.infos) == 1
        assert len(result.warnings) == 1
        assert len(result.errors) == 1