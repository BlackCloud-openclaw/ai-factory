"""测试 EventUpcaster"""
from src.writing.causality.upcaster import EventUpcaster, LATEST_EVENT_SCHEMA_VERSION


def test_upcaster_adds_semantic():
    old_event = {
        "type": "item_acquire",
        "actor": "LinYi",
        "item": "Sword",
        "event_schema_version": 1
    }
    upgraded = EventUpcaster.upcast(old_event, version=1)
    assert "semantic" in upgraded
    assert upgraded["semantic"] in ("state_mutation", "observation", "dream")
    assert upgraded["event_schema_version"] == LATEST_EVENT_SCHEMA_VERSION


def test_upcaster_preserves_existing_semantic():
    event = {
        "type": "dialogue",
        "semantic": "dialogue",
        "event_schema_version": 1
    }
    upgraded = EventUpcaster.upcast(event, version=1)
    assert upgraded["semantic"] == "dialogue"


def test_upcaster_no_version_change_for_latest():
    event = {
        "type": "realm_upgrade",
        "semantic": "state_mutation",
        "event_schema_version": LATEST_EVENT_SCHEMA_VERSION
    }
    upgraded = EventUpcaster.upcast(event, LATEST_EVENT_SCHEMA_VERSION)
    assert upgraded == event  # 无变化