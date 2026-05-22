import pytest
from src.writing.causality.rule_engine import RuleEngine, Rule, Precondition
from src.writing.causality.predicate import Predicate


def test_rule_loading():
    engine = RuleEngine()
    assert len(engine.rules) >= 3
    assert engine.get_rules_for_event("use_item")[0].id == "use_item_requires_ownership"


def test_match_preconditions_success():
    engine = RuleEngine()
    rule = engine.get_rules_for_event("use_item")[0]
    predicates = {
        Predicate(subject="LinYi", relation="has_item", object="Sword",
                  confidence=1.0, source_event_type="item_acquire").identity_key():
        Predicate(subject="LinYi", relation="has_item", object="Sword",
                  confidence=1.0, source_event_type="item_acquire")
    }
    matched, bindings, missing = engine.match_preconditions(rule, predicates)
    assert matched is True
    assert bindings["actor"] == "LinYi"
    assert bindings["item"] == "Sword"


def test_match_preconditions_missing():
    engine = RuleEngine()
    rule = engine.get_rules_for_event("use_item")[0]
    predicates = {}  # 没有 has_item
    matched, bindings, missing = engine.match_preconditions(rule, predicates)
    assert matched is False
    assert len(missing) > 0