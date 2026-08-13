# tests/unit/surfaces/test_compatibility.py

import pytest
from dataclasses import replace

from src.surfaces.definition import PatternDefinition, ObservationSpec
from src.surfaces.compatibility import (
    upgrade_pattern,
    upgrade_observation,
    is_upgraded,
    SurfaceCompatibilityError,
)
from src.capabilities import CapabilityRef


def test_upgrade_keyword():
    pattern = PatternDefinition(
        name="test",
        matcher="keyword",
        config={"keywords": ["a", "b"]},
    )
    upgraded = upgrade_pattern(pattern)

    assert upgraded.capability_ref == CapabilityRef.parse("builtin.keyword")
    assert upgraded.matcher is None
    assert upgraded.name == "test"
    assert upgraded.config == {"keywords": ["a", "b"]}


def test_upgrade_regex():
    pattern = PatternDefinition(
        name="test",
        matcher="regex",
        config={"pattern": r"\d+"},
    )
    upgraded = upgrade_pattern(pattern)

    assert upgraded.capability_ref == CapabilityRef.parse("builtin.regex")
    assert upgraded.matcher is None


def test_upgrade_quotation():
    pattern = PatternDefinition(
        name="test",
        matcher="quotation",
        config={},
    )
    upgraded = upgrade_pattern(pattern)

    assert upgraded.capability_ref == CapabilityRef.parse("builtin.quotation")
    assert upgraded.matcher is None


def test_upgrade_already_has_capability():
    pattern = PatternDefinition(
        name="test",
        capability_ref=CapabilityRef.parse("builtin.keyword"),
        config={},
        matcher=None,
    )
    upgraded = upgrade_pattern(pattern)
    assert upgraded == pattern


def test_upgrade_idempotent():
    pattern = PatternDefinition(
        name="test",
        matcher="keyword",
        config={},
    )
    p1 = upgrade_pattern(pattern)
    p2 = upgrade_pattern(p1)

    assert p1 == p2
    assert is_upgraded(p1)
    assert is_upgraded(p2)


def test_upgrade_unknown_matcher():
    pattern = PatternDefinition(
        name="test",
        matcher="unknown",
        config={},
    )
    with pytest.raises(SurfaceCompatibilityError) as exc:
        upgrade_pattern(pattern)
    assert "unknown" in str(exc.value)


def test_upgrade_observation():
    p1 = PatternDefinition(name="a", matcher="keyword", config={})
    p2 = PatternDefinition(name="b", matcher="quotation", config={})
    obs = ObservationSpec(patterns=(p1, p2))

    upgraded_obs = upgrade_observation(obs)

    assert len(upgraded_obs.patterns) == 2
    assert upgraded_obs.patterns[0].capability_ref == CapabilityRef.parse("builtin.keyword")
    assert upgraded_obs.patterns[1].capability_ref == CapabilityRef.parse("builtin.quotation")
    assert all(p.matcher is None for p in upgraded_obs.patterns)


def test_upgrade_observation_idempotent():
    p1 = PatternDefinition(name="a", matcher="keyword", config={})
    obs = ObservationSpec(patterns=(p1,))
    o1 = upgrade_observation(obs)
    o2 = upgrade_observation(o1)
    assert o1 == o2