import pytest
from src.writing.causality.predicate import Predicate, normalize_object

def test_predicate_identity():
    p1 = Predicate(subject="LinYi", relation="realm", object="Foundation")
    p2 = Predicate(subject="linyi", relation="realm", object="foundation")
    assert p1.identity_key() == p2.identity_key()

def test_predicate_negated():
    p1 = Predicate(subject="LinYi", relation="has_item", object="Sword", negated=False)
    p2 = Predicate(subject="LinYi", relation="has_item", object="Sword", negated=True)
    assert p1.identity_key() != p2.identity_key()

def test_normalize_object():
    assert normalize_object({"a": 1, "b": 2}) == '{"a":1,"b":2}'
    assert normalize_object("  Foundation  ") == "foundation"