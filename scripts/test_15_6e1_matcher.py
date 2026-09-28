#!/usr/bin/env python
"""
15.6-E1 Matcher Regression Test

测试 location_change 的 location_enter 别名归一化。
确保 plot_flag 和 knowledge_gain 的回归测试仍然通过。
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# 从 phase15_5_stage_b_corrected.py 复制的必要定义
# ============================================================

STAGE_B_EVENT_TYPE_ALIASES = {
    "plot_flag": {"plot_flag", "plot_flag_set"},
    "realm_change": {"realm_change", "realm_upgrade"},
    "inventory_acquire": {"inventory_acquire", "item_acquire"},
    "location_change": {"location_change", "location_enter"},
    "relationship_change": {"relationship_change"},
    "knowledge_gain": {"knowledge_gain", "discovery"},
}

FIELD_ALIASES = {
    "flag": "name",
    "to_location": "location",
    "knowledge": "name",
}

def normalize_value(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return v.strip()
    return v

def values_equal(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return normalize_value(a) == normalize_value(b)

# ============================================================
# _get_event_identity_value（包含 plot_flag 和 knowledge_gain）
# ============================================================
def _get_event_identity_value(req_type: str, event: dict, field: str):
    if req_type == "plot_flag" and field == "name":
        if event.get("name") is not None:
            return event["name"]
        if event.get("flag") is not None:
            return event["flag"]
        return None

    if req_type == "knowledge_gain" and field == "name":
        if event.get("name") is not None:
            return event["name"]
        if event.get("discovery") is not None:
            return event["discovery"]
        return None

    return event.get(field)

# ============================================================
# 模拟 StateChange 对象
# ============================================================
class MockStateChange:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

# ============================================================
# match_requirement_to_events
# ============================================================
def match_requirement_to_events(
    sc,
    events: List[Dict],
    used_indices: set,
) -> Tuple[Dict, Optional[int]]:
    req_type = sc.type
    allowed_types = STAGE_B_EVENT_TYPE_ALIASES.get(req_type, {req_type})

    candidate_indices = [
        i for i, evt in enumerate(events)
        if i not in used_indices
        and evt.get("type") in allowed_types
    ]

    if not candidate_indices:
        return {
            "status": "NO_EVENT",
            "reason": "No event with type matching",
            "matched_event": None,
        }, None

    identity_fields_map = {
        "plot_flag": ["name", "value"],
        "realm_change": ["actor", "to_major_realm", "to_minor_stage"],
        "inventory_acquire": ["actor", "item"],
        "location_change": ["actor", "location", "to_location"],
        "relationship_change": ["from_char", "to_char"],
        "knowledge_gain": ["name"],
    }

    identity_fields = identity_fields_map.get(req_type, [])

    for idx in candidate_indices:
        evt = events[idx]
        identity_ok = True
        for f in identity_fields:
            contract_field = f
            writer_field = f
            if f in FIELD_ALIASES:
                writer_field = f
            req_val = getattr(sc, contract_field, None)
            evt_val = _get_event_identity_value(req_type, evt, writer_field)
            if not values_equal(req_val, evt_val):
                identity_ok = False
                break
        if identity_ok:
            return {
                "status": "IDENTITY_MATCH",
                "reason": "All identity fields match",
                "matched_event": evt,
            }, idx

    if candidate_indices:
        evt = events[candidate_indices[0]]
        return {
            "status": "TYPE_MATCH",
            "reason": "Type matches but identity fields differ",
            "matched_event": evt,
        }, candidate_indices[0]

    return {
        "status": "NO_EVENT",
        "reason": "No event with type matching",
        "matched_event": None,
    }, None


# ============================================================
# 测试执行
# ============================================================
def test_plot_flag_regression():
    print("\n📊 测试 plot_flag 回归...")
    sc = MockStateChange(type="plot_flag", name="A", value=True)

    cases = [
        ({"type": "plot_flag_set", "name": "A", "value": True}, "IDENTITY_MATCH"),
        ({"type": "plot_flag_set", "flag": "A", "value": True}, "IDENTITY_MATCH"),
        ({"type": "plot_flag_set", "flag": "B", "value": True}, "TYPE_MATCH"),
        ({"type": "plot_flag_set", "flag": "A", "value": False}, "TYPE_MATCH"),
        ({"type": "plot_flag_set", "value": True}, "TYPE_MATCH"),
        ({"type": "plot_flag_set", "flag": "C"}, "TYPE_MATCH"),
    ]

    for event, expected in cases:
        result, _ = match_requirement_to_events(sc, [event], set())
        status = result["status"]
        assert status == expected, f"plot_flag regression fail: expected {expected}, got {status}"
    print("  ✅ plot_flag 回归全部通过")


def test_knowledge_gain_regression():
    print("\n📊 测试 knowledge_gain 回归...")
    sc = MockStateChange(type="knowledge_gain", name="法则熔炉奥义", value=True)

    cases = [
        ({"type": "knowledge_gain", "name": "法则熔炉奥义"}, "IDENTITY_MATCH"),
        ({"type": "discovery", "discovery": "法则熔炉奥义"}, "IDENTITY_MATCH"),
        ({"type": "discovery", "discovery": "别的知识"}, "TYPE_MATCH"),
        ({"type": "discovery", "value": True}, "TYPE_MATCH"),
        ({"type": "plot_flag_set", "flag": "法则熔炉奥义"}, "NO_EVENT"),
    ]

    for event, expected in cases:
        result, _ = match_requirement_to_events(sc, [event], set())
        status = result["status"]
        assert status == expected, f"knowledge_gain fail: expected {expected}, got {status}"
    print("  ✅ knowledge_gain 回归全部通过")


def test_location_change_normalization():
    print("\n📊 测试 location_change 归一化...")
    sc = MockStateChange(type="location_change", actor="林逸", location="道观藏经阁")

    cases = [
        # Case 1: location_change 直接匹配 → IDENTITY_MATCH
        ({"type": "location_change", "actor": "林逸", "location": "道观藏经阁"}, "IDENTITY_MATCH"),
        # Case 2: location_enter 匹配 → IDENTITY_MATCH (15.6-E1 核心)
        ({"type": "location_enter", "actor": "林逸", "location": "道观藏经阁"}, "IDENTITY_MATCH"),
        # Case 3: location_enter 不匹配 → TYPE_MATCH
        ({"type": "location_enter", "actor": "林逸", "location": "别处"}, "TYPE_MATCH"),
        # Case 4: 无 location → TYPE_MATCH
        ({"type": "location_enter", "actor": "林逸"}, "TYPE_MATCH"),
        # Case 5: 错误类型 → NO_EVENT
        ({"type": "plot_flag_set", "flag": "道观藏经阁"}, "NO_EVENT"),
    ]

    for event, expected in cases:
        result, _ = match_requirement_to_events(sc, [event], set())
        status = result["status"]
        assert status == expected, f"location_change fail: expected {expected}, got {status}"
    print("  ✅ location_change 归一化全部通过")


def run_all_tests():
    print("=" * 60)
    print("15.6-E1 Matcher Regression Test")
    print("=" * 60)
    test_plot_flag_regression()
    test_knowledge_gain_regression()
    test_location_change_normalization()
    print("\n" + "=" * 60)
    print("✅ 所有测试通过")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()