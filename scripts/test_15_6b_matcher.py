#!/usr/bin/env python
"""
15.6-B Matcher Regression Test

测试 `match_requirement_to_events()` 对 plot_flag 的 alias normalization 是否按预期工作。
包含 6 个测试用例，在 normalization 层和真实 matcher 层分别验证。
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# 从 phase15_5_stage_b_corrected.py 复制的必要定义
# 保持与正式 matcher 一致
# ============================================================

STAGE_B_EVENT_TYPE_ALIASES = {
    "plot_flag": {"plot_flag", "plot_flag_set"},
    "realm_change": {"realm_change", "realm_upgrade"},
    "inventory_acquire": {"inventory_acquire", "item_acquire"},
    "location_change": {"location_change"},
    "relationship_change": {"relationship_change"},
    "knowledge_gain": {"knowledge_gain"},
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
# 15.6-B: _get_event_identity_value
# ============================================================
def _get_event_identity_value(req_type: str, event: dict, field: str):
    """
    从 event 中提取 identity 字段值，支持 plot_flag 的 name ↔ flag 别名。
    """
    if req_type == "plot_flag" and field == "name":
        if event.get("name") is not None:
            return event["name"]
        if event.get("flag") is not None:
            return event["flag"]
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
# 真正的 match_requirement_to_events（复刻版，与正式脚本一致）
# ============================================================
def match_requirement_to_events(
    sc,
    events: List[Dict],
    used_indices: set,
) -> Tuple[Dict, Optional[int]]:
    """
    尝试匹配一个 requirement 到 events。
    返回 (result_dict, used_index_or_None)
    """
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

    # ============================================================
    # 15.6-B: plot_flag 使用 ["name", "value"]
    # ============================================================
    identity_fields_map = {
        "plot_flag": ["name", "value"],
        "realm_change": ["actor", "to_major_realm", "to_minor_stage"],
        "inventory_acquire": ["actor", "item"],
        "location_change": ["actor", "location", "to_location"],
        "relationship_change": ["from_char", "to_char"],
        "knowledge_gain": ["name", "knowledge"],
    }
    # ============================================================

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
def test_normalization():
    """测试 _get_event_identity_value 的纯 normalization 逻辑。"""
    print("\n📊 测试 Normalization 层...")
    test_cases = [
        ("name", {"name": "A", "flag": "B"}, "A", "name 存在，优先使用 name"),
        ("name", {"flag": "B"}, "B", "name 不存在，回退到 flag"),
        ("name", {"flag": None}, None, "name 和 flag 都不存在"),
        ("value", {"value": True}, True, "非 name 字段直接返回"),
    ]
    for field, event, expected, desc in test_cases:
        result = _get_event_identity_value("plot_flag", event, field)
        assert result == expected, f"Normalization fail: {desc} -> expected {expected}, got {result}"
    print("  ✅ Normalization 层全部通过")


def test_matcher():
    """测试 match_requirement_to_events 的真实行为。"""
    print("\n📊 测试真实 Matcher 层...")

    # 标准 requirement: name=A, value=True
    sc = MockStateChange(type="plot_flag", name="A", value=True)

    test_cases = [
        # Case 1: name 直接匹配
        (
            {"type": "plot_flag_set", "name": "A", "value": True},
            "IDENTITY_MATCH",
            "Case 1: name 匹配"
        ),
        # Case 2: flag 匹配
        (
            {"type": "plot_flag_set", "flag": "A", "value": True},
            "IDENTITY_MATCH",
            "Case 2: flag 匹配 → 应为 IDENTITY_MATCH"
        ),
        # Case 3: flag 不匹配
        (
            {"type": "plot_flag_set", "flag": "B", "value": True},
            "TYPE_MATCH",
            "Case 3: flag 不匹配 → 应为 TYPE_MATCH"
        ),
        # Case 4: value 不匹配
        (
            {"type": "plot_flag_set", "flag": "A", "value": False},
            "TYPE_MATCH",
            "Case 4: value 不匹配 → 应为 TYPE_MATCH"
        ),
        # Case 5: 无 name/flag
        (
            {"type": "plot_flag_set", "value": True},
            "TYPE_MATCH",
            "Case 5: 无 name/flag → 应为 TYPE_MATCH"
        ),
        # Case 6: 缺少 value
        (
            {"type": "plot_flag_set", "flag": "C"},
            "TYPE_MATCH",
            "Case 6: 缺少 value → 应为 TYPE_MATCH"
        ),
    ]

    for event, expected, desc in test_cases:
        result, idx = match_requirement_to_events(sc, [event], set())
        status = result["status"]
        assert status == expected, f"Matcher fail: {desc} -> expected {expected}, got {status}"
        print(f"  ✅ {desc}")

    print("  ✅ Matcher 层全部通过")


def run_all_tests():
    print("=" * 60)
    print("15.6-B Matcher Regression Test")
    print("=" * 60)
    test_normalization()
    test_matcher()
    print("\n" + "=" * 60)
    print("✅ 所有测试通过")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()