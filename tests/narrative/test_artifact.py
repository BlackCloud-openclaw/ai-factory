import pytest
from dataclasses import FrozenInstanceError
from uuid import UUID, uuid4

from src.narrative.artifact import NarrativeArtifact


class TestNarrativeArtifact:
    def test_immutable(self):
        artifact = NarrativeArtifact("hello")
        with pytest.raises(FrozenInstanceError):
            artifact.text = "world"  # type: ignore

    def test_empty_text(self):
        artifact = NarrativeArtifact("")
        assert artifact.text == ""

    def test_to_dict_returns_dict(self):
        artifact = NarrativeArtifact("test")
        result = artifact.to_dict()
        assert "text" in result
        assert "artifact_id" in result
        assert "schema_version" in result
        assert isinstance(result, dict)

    def test_serialization_roundtrip(self):
        original = NarrativeArtifact("test content")
        data = original.to_dict()
        restored = NarrativeArtifact.from_dict(data)

        assert original.text == restored.text
        assert str(original.artifact_id) == str(restored.artifact_id)

    def test_uuid_generation(self):
        a1 = NarrativeArtifact("a")
        a2 = NarrativeArtifact("b")
        assert a1.artifact_id != a2.artifact_id
        assert isinstance(a1.artifact_id, UUID)

    def test_uuid_tolerance(self):
        u = uuid4()
        a1 = NarrativeArtifact.from_dict({"text": "a", "artifact_id": str(u)})
        a2 = NarrativeArtifact.from_dict({"text": "a", "artifact_id": u})
        assert a1.artifact_id == a2.artifact_id

    def test_from_dict_missing_id_generates_uuid(self):
        data = {"text": "hello"}
        artifact = NarrativeArtifact.from_dict(data)
        assert artifact.artifact_id is not None
        assert isinstance(artifact.artifact_id, UUID)
        assert str(artifact.artifact_id) != "00000000-0000-0000-0000-000000000000"

    def test_schema_version(self):
        assert NarrativeArtifact.SCHEMA_VERSION == "1.0.0"