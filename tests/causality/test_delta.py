"""测试 PredicateDelta 结构"""
from src.writing.causality.delta import PredicateDelta, PredicateRef
from src.writing.causality.predicate import Predicate


def test_delta_creation():
    delta = PredicateDelta(
        novel_id="test_novel",
        event_id=101,
        projection_version=1,
        event_semantic="state_mutation",
        to_activate=[
            Predicate(subject="LinYi", relation="has_item", object="Sword")
        ],
        to_deactivate=[
            PredicateRef(identity_key="LinYi|has_item|old_sword", event_id=101)
        ]
    )
    assert not delta.is_empty()
    assert len(delta.to_activate) == 1
    assert len(delta.to_deactivate) == 1


def test_empty_delta():
    delta = PredicateDelta(
        novel_id="test_novel",
        event_id=102,
        projection_version=1,
        event_semantic="dialogue"
    )
    assert delta.is_empty()