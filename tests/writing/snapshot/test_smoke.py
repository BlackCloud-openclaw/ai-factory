# tests/writing/snapshot/test_smoke.py

import tempfile
from pathlib import Path

import pytest

from src.writing.snapshot.serializer import JsonSerializer
from src.writing.snapshot.writer import SnapshotWriter
from src.writing.snapshot.loader import SnapshotLoader
from tests.fixtures.builders.snapshot_builder import build_sample_snapshot


def test_smoke_full_pipeline():
    """验证整个 Snapshot Runtime 全链路"""
    snapshot = build_sample_snapshot()
    serializer = JsonSerializer()

    # 1. 编码
    data1 = serializer.serialize(snapshot)

    # 2. 解码
    restored1 = serializer.deserialize(data1)

    # 3. 重新编码
    data2 = serializer.serialize(restored1)

    # 4. 验证一致性
    assert data1 == data2, "Double-serialize mismatch"


def test_smoke_writer_loader():
    """验证 Writer + Loader 全链路"""
    snapshot = build_sample_snapshot()

    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        writer = SnapshotWriter(base_dir)
        loader = SnapshotLoader()

        file_path = writer.write(snapshot)
        loaded = loader.load(file_path)

        assert snapshot == loaded, "Writer/Loader round-trip failed"


def test_smoke_sha256_stable():
    """验证 SHA256 在多次序列化中稳定"""
    snapshot = build_sample_snapshot()
    serializer = JsonSerializer()

    hashes = []
    for _ in range(3):
        data = serializer.serialize(snapshot)
        import hashlib
        hashes.append(hashlib.sha256(data).hexdigest())

    assert hashes[0] == hashes[1] == hashes[2], "SHA256 not stable"