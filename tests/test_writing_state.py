import pytest
from src.writing.events import Event, EVENT_CHARACTER_UPDATE
from src.writing.reducer import apply_event, reduce_state
from src.writing.invariants import validate_event

def test_apply_event():
    state = {}
    event = Event.new(EVENT_CHARACTER_UPDATE, {"name": "林风", "updates": {"realm": "筑基"}}, novel_id="test")
    new_state = apply_event(state, event)
    assert new_state["characters"]["林风"]["realm"] == "筑基"

def test_reduce_state():
    events = [
        Event.new(EVENT_CHARACTER_UPDATE, {"name": "林风", "updates": {"realm": "炼气"}}, novel_id="test"),
        Event.new(EVENT_CHARACTER_UPDATE, {"name": "林风", "updates": {"realm": "筑基"}}, novel_id="test"),
    ]
    # 手动分配 sequence_id（模拟数据库自增）
    for i, e in enumerate(events, 1):
        e.sequence_id = i
    state = reduce_state(events)
    assert state["characters"]["林风"]["realm"] == "筑基"

def test_validate_event_realm_skip():
    state = {"characters": {"林风": {"realm": "筑基"}}}
    event = Event.new(EVENT_CHARACTER_UPDATE, {"name": "林风", "updates": {"realm": "元婴"}}, novel_id="test")
    ok, msg = validate_event(state, event)
    assert not ok
    assert "跨级" in msg