import pytest
from src.writing.causality.projector import DeltaEngine
from src.writing.causality.predicate import Predicate

def test_item_acquire_produces_predicate():
    engine = DeltaEngine()
    event = {
        'novel_id': 'test',
        'id': 101,
        'type': 'item_acquire',
        'semantic': 'state_mutation',
        'actor': 'LinYi',
        'item': 'JadePendant'
    }
    delta = engine.compute_delta({}, event)
    assert len(delta.to_activate) == 1
    assert delta.to_activate[0].subject == 'LinYi'
    assert delta.to_activate[0].relation == 'has_item'
    assert delta.to_activate[0].object == 'JadePendant'

def test_dream_event_lowers_confidence():
    engine = DeltaEngine()
    event = {
        'novel_id': 'test',
        'id': 102,
        'type': 'item_acquire',
        'semantic': 'dream',
        'actor': 'LinYi',
        'item': 'MysticSword'
    }
    delta = engine.compute_delta({}, event)
    assert delta.to_activate[0].confidence == 0.4
    assert delta.to_activate[0].priority == 'flavor'