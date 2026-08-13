import pytest
from dataclasses import FrozenInstanceError
from uuid import UUID, uuid4

from src.narrative.snapshot import StorySnapshot


class TestStorySnapshot:
    def test_immutable(self):
        snapshot = StorySnapshot()
        with pytest.raises(FrozenInstanceError):
            snapshot.snapshot_id = None  # type: ignore

    def test_empty_projection(self):
        snapshot = StorySnapshot()
        assert snapshot.projection == {}

    def test_any_projection(self):
        snapshot = StorySnapshot(projection={"events": ["e1"]})
        assert snapshot.projection["events"] == ["e1"]

        snapshot2 = StorySnapshot(projection=["event_a", "event_b"])
        assert len(snapshot2.projection) == 2

    def test_to_dict_returns_dict(self):
        snapshot = StorySnapshot()
        result = snapshot.to_dict()
        assert "snapshot_id" in result
        assert "projection" in result
        assert "schema_version" in result
        assert isinstance(result, dict)

    def test_serialization_roundtrip(self):
        original = StorySnapshot(
            projection={"events": ["event_1"], "timeline": ["step_a"]}
        )
        data = original.to_dict()
        restored = StorySnapshot.from_dict(data)

        assert str(original.snapshot_id) == str(restored.snapshot_id)
        assert original.projection == restored.projection

    def test_uuid_generation(self):
        s1 = StorySnapshot()
        s2 = StorySnapshot()
        assert s1.snapshot_id != s2.snapshot_id
        assert isinstance(s1.snapshot_id, UUID)

    def test_uuid_tolerance(self):
        u = uuid4()
        s1 = StorySnapshot.from_dict({"snapshot_id": str(u)})
        s2 = StorySnapshot.from_dict({"snapshot_id": u})
        assert s1.snapshot_id == s2.snapshot_id

    def test_from_dict_missing_id_generates_uuid(self):
        data = {"projection": {}}
        snapshot = StorySnapshot.from_dict(data)
        assert snapshot.snapshot_id is not None
        assert isinstance(snapshot.snapshot_id, UUID)
        assert str(snapshot.snapshot_id) != "00000000-0000-0000-0000-000000000000"

    def test_schema_version(self):
        assert StorySnapshot.SCHEMA_VERSION == "1.0.0"