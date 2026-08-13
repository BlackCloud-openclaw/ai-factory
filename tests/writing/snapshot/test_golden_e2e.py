# tests/writing/snapshot/test_golden_e2e.py

import hashlib
from pathlib import Path

import pytest

from src.writing.snapshot.serializer import JsonSerializer
from src.writing.snapshot.loader import SnapshotLoader
from tests.fixtures.builders.snapshot_builder import build_sample_snapshot


FIXTURES_DIR = Path("tests/fixtures/snapshots")


def test_golden_deserialize_serialize_match():
    """Golden → deserialize → serialize → SHA256 必须与原始 Golden 一致"""
    golden_path = FIXTURES_DIR / "v1.0" / "canonical.json"
    if not golden_path.exists():
        pytest.skip("Golden file not found, run scripts/generate_golden.py first")

    original_data = golden_path.read_bytes()
    original_sha256 = hashlib.sha256(original_data).hexdigest()

    loader = SnapshotLoader()
    snapshot = loader.load(golden_path)

    serializer = JsonSerializer()
    regenerated_data = serializer.serialize(snapshot)
    regenerated_sha256 = hashlib.sha256(regenerated_data).hexdigest()

    # 端到端不可变性验证
    assert original_sha256 == regenerated_sha256, (
        f"Golden file SHA256 mismatch!\n"
        f"Original:  {original_sha256}\n"
        f"Regenerated: {regenerated_sha256}"
    )