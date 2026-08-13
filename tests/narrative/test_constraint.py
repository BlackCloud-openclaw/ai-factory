import pytest
from uuid import UUID, uuid4

from src.narrative.constraint import NarrativeConstraint


class TestNarrativeConstraint:
    def test_serialization_roundtrip(self):
        original = NarrativeConstraint(
            payload={"events": ["e1", "e2"], "flags": {"flag": True}}
        )
        data = original.to_dict()
        restored = NarrativeConstraint.from_dict(data)

        assert str(original.constraint_id) == str(restored.constraint_id)
        assert original.payload == restored.payload

    def test_empty_constraint(self):
        c = NarrativeConstraint()
        assert c.payload == {}

    def test_any_payload(self):
        c = NarrativeConstraint(payload={"key": "value", "nested": {"a": 1}})
        assert c.payload["nested"]["a"] == 1

    def test_uuid_tolerance(self):
        u = uuid4()
        c1 = NarrativeConstraint.from_dict({"constraint_id": str(u)})
        c2 = NarrativeConstraint.from_dict({"constraint_id": u})
        assert c1.constraint_id == c2.constraint_id

    def test_from_dict_missing_id_generates_uuid(self):
        data = {"payload": {}}
        c = NarrativeConstraint.from_dict(data)
        assert c.constraint_id is not None
        assert isinstance(c.constraint_id, UUID)
        assert str(c.constraint_id) != "00000000-0000-0000-0000-000000000000"

    def test_schema_version(self):
        assert NarrativeConstraint.SCHEMA_VERSION == "1.0.0"