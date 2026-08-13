# tests/unit/snapshot/runtime/test_id.py
"""
B3.1: SnapshotId 单元测试
"""

import pytest
from uuid import UUID

from src.writing.snapshot.runtime.id import SnapshotId


class TestSnapshotId:
    def test_new_generates_unique_ids(self):
        id1 = SnapshotId.new()
        id2 = SnapshotId.new()
        assert id1 != id2
        assert str(id1) != str(id2)

    def test_from_uuid_accepts_uuid_object(self):
        uuid_obj = UUID("12345678-1234-5678-1234-567812345678")
        sid = SnapshotId.from_uuid(uuid_obj)
        assert sid.value == uuid_obj
        assert str(sid) == "12345678-1234-5678-1234-567812345678"

    def test_from_string_valid_uuid(self):
        uuid_str = "12345678-1234-5678-1234-567812345678"
        sid = SnapshotId.from_string(uuid_str)
        assert str(sid) == uuid_str
        assert sid.value == UUID(uuid_str)

    def test_from_string_invalid_uuid_raises(self):
        with pytest.raises(ValueError, match="Invalid SnapshotId"):
            SnapshotId.from_string("abc")

        with pytest.raises(ValueError, match="Invalid SnapshotId"):
            SnapshotId.from_string("12345678-1234-5678-1234-56781234567")

    def test_round_trip_consistency(self):
        sid = SnapshotId.new()
        as_str = str(sid)
        reconstructed = SnapshotId.from_string(as_str)
        assert sid == reconstructed
        assert sid.value == reconstructed.value

    def test_hashable(self):
        sid = SnapshotId.new()
        d = {sid: "value"}
        assert d[sid] == "value"
        assert sid in d

    def test_equality_based_on_uuid(self):
        uuid_str = "12345678-1234-5678-1234-567812345678"
        id1 = SnapshotId.from_string(uuid_str)
        id2 = SnapshotId.from_string(uuid_str)
        assert id1 == id2
        assert hash(id1) == hash(id2)