# tests/causality/test_rule_engine.py 内容替换为：
import pytest
from src.writing.causality.rule_engine import RuleEngine
from src.writing.causality.predicate import Predicate

def test_match_preconditions_success():
    engine = RuleEngine()
    rules = engine.get_rules_for_event("use_item")
    if not rules:
        pytest.skip("No rules for use_item")
    rule = rules[0]
    predicates = {
        Predicate(subject="LinYi", relation="has_item", object="Sword",
                  confidence=1.0, source_event_type="item_acquire").identity_key():
        Predicate(subject="LinYi", relation="has_item", object="Sword",
                  confidence=1.0, source_event_type="item_acquire")
    }
    event = {"type": "use_item", "actor": "LinYi", "item": "Sword"}
    matched, bindings, missing = engine.match_rule_with_event(rule, predicates, event)
    assert matched is True
    assert bindings.get("actor") == "linyi"  # normalized
    assert missing == []

def test_match_preconditions_missing():
    engine = RuleEngine()
    rules = engine.get_rules_for_event("use_item")
    if not rules:
        pytest.skip("No rules for use_item")
    rule = rules[0]
    predicates = {}  # 没有 has_item
    event = {"type": "use_item", "actor": "LinYi", "item": "Sword"}
    matched, bindings, missing = engine.match_rule_with_event(rule, predicates, event)
    assert matched is False
    assert len(missing) > 0