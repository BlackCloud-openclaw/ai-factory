import pytest
from uuid import UUID, uuid4

from src.narrative import (
    StorySnapshot,
    NarrativeContext,
    ChapterMetadata,
    CharacterArc,
    ArcStatus,
)


class TestNarrativeContext:
    def test_serialization_roundtrip(self):
        snapshot = StorySnapshot(projection={"events": ["e1"]})
        meta = ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3)

        original = NarrativeContext(
            story=snapshot,
            metadata=meta,
            character_arcs={
                "protagonist": CharacterArc(
                    character_id="protagonist",
                    arc_id="arc_1",
                    status=ArcStatus.OPEN,
                    progress=0.5,
                )
            },
        )

        data = original.to_dict()
        restored = NarrativeContext.from_dict(data)

        assert str(original.context_id) == str(restored.context_id)
        assert original.metadata.volume == restored.metadata.volume
        assert original.character_arcs["protagonist"].progress == 0.5
        assert original.character_arcs["protagonist"].status == ArcStatus.OPEN

    def test_empty_context(self):
        snapshot = StorySnapshot(projection={})
        meta = ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3)
        ctx = NarrativeContext(story=snapshot, metadata=meta)

        assert ctx.previous_chapters == ()
        assert ctx.character_arcs == {}

    def test_previous_chapters_immutable(self):
        snapshot = StorySnapshot(projection={})
        meta = ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3)
        ctx = NarrativeContext(
            story=snapshot,
            metadata=meta,
            previous_chapters=("ch1", "ch2"),
        )

        with pytest.raises(TypeError):
            ctx.previous_chapters[0] = "ch3"  # type: ignore

    def test_uuid_tolerance(self):
        u = uuid4()
        snapshot = StorySnapshot(projection={})
        meta = ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3)

        data = {
            "context_id": str(u),
            "story": snapshot.to_dict(),
            "metadata": {
                "volume": 1,
                "chapter": 1,
                "scene_index": 0,
                "total_scenes": 3,
            },
        }

        ctx = NarrativeContext.from_dict(data)
        assert ctx.context_id == u

    def test_from_dict_missing_id_generates_uuid(self):
        snapshot = StorySnapshot(projection={})
        meta = ChapterMetadata(volume=1, chapter=1, scene_index=0, total_scenes=3)

        data = {
            "story": snapshot.to_dict(),
            "metadata": {
                "volume": 1,
                "chapter": 1,
                "scene_index": 0,
                "total_scenes": 3,
            },
        }

        ctx = NarrativeContext.from_dict(data)
        assert ctx.context_id is not None
        assert isinstance(ctx.context_id, UUID)
        assert str(ctx.context_id) != "00000000-0000-0000-0000-000000000000"

    def test_arc_status_enum(self):
        assert ArcStatus.OPEN.value == "open"
        assert ArcStatus.RESOLVED.value == "resolved"
        assert ArcStatus.ABANDONED.value == "abandoned"

    def test_schema_version(self):
        assert NarrativeContext.SCHEMA_VERSION == "1.0.0"
        assert ChapterMetadata.SCHEMA_VERSION == "1.0.0"
        assert CharacterArc.SCHEMA_VERSION == "1.0.0"