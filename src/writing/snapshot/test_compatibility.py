# tests/writing/snapshot/test_compatibility.py

import hashlib
from pathlib import Path

import pytest

from src.writing.snapshot.loader import SnapshotLoader


FIXTURES_DIR = Path("tests/fixtures/snapshots")


@pytest.fixture
def golden_paths():
    return list(FIXTURES_DIR.glob("*/canonical.json"))


def test_golden_hash_matches(golden_paths):
    for path in golden_paths:
        data = path.read_bytes()
        sha256_path = path.parent / "snapshot.sha256"
        expected = sha256_path.read_text().strip()
        actual = hashlib.sha256(data).hexdigest()
        assert actual == expected, f"Hash mismatch for {path}"


def test_golden_deserializes(golden_paths):
    loader = SnapshotLoader()
    for path in golden_paths:
        snapshot = loader.load(path)
        assert snapshot.identity.snapshot_id is not None
        assert snapshot.manifest.schema_version is not None