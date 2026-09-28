#!/usr/bin/env python
"""
Phase 15.5/15.6: Field Semantic Diagnosis (Corrected Measurement)

对 Treatment 样本进行字段级匹配分析。
分层诊断：TYPE → IDENTITY → VALUE

包含修正:
- 15.6-B: plot_flag name ↔ flag 归一化
- 15.6-D1: knowledge_gain name ↔ discovery 归一化
"""

import sys
import asyncio
import asyncpg
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.writing.planning_contract import PlanningContract

DSN = "postgresql://woami:kali@localhost:5432/ai_factory"

EXPECTED_TREATMENT_SAMPLES = 56

# ============================================================
# Stage B 独立维护的类型别名（与 Writer 输出一致）
# ============================================================
STAGE_B_EVENT_TYPE_ALIASES = {
    "plot_flag": {"plot_flag", "plot_flag_set"},
    "realm_change": {"realm_change", "realm_upgrade"},
    "inventory_acquire": {"inventory_acquire", "item_acquire"},
    "location_change": {"location_change", "location_enter"},  # ✅ 新增
    "relationship_change": {"relationship_change"},
    "knowledge_gain": {"knowledge_gain", "discovery"},  # 15.6-D1
}

# ============================================================
# 字段别名映射：Writer 字段名 → Contract 字段名
# ============================================================
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
# 15.6-B + 15.6-D1: _get_event_identity_value
# ============================================================
def _get_event_identity_value(req_type: str, event: dict, field: str):
    """
    从 event 中提取 identity 字段值，支持字段别名归一化。
    
    - plot_flag: name ↔ flag (15.6-B)
    - knowledge_gain: name ↔ discovery (15.6-D1)
    """
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
# 匹配引擎（按 requirement 粒度）
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

    # 找所有类型匹配的事件
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

    # 定义每个 type 的 identity 字段组合
    identity_fields_map = {
        "plot_flag": ["name", "value"],                     # 15.6-B: 修正为 name + value
        "realm_change": ["actor", "to_major_realm", "to_minor_stage"],
        "inventory_acquire": ["actor", "item"],
        "location_change": ["actor", "location", "to_location"],
        "relationship_change": ["from_char", "to_char"],
        "knowledge_gain": ["name"],                         # 15.6-D1: 只保留 name
    }

    identity_fields = identity_fields_map.get(req_type, [])

    # 先找 identity 完全匹配
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

    # 如果没有 identity 完全匹配，检查是否 type 匹配但字段值不同
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


async def stage_b_diagnose():
    conn = await asyncpg.connect(DSN)

    rows = await conn.fetch("""
        SELECT
            scene_id,
            contract_data,
            writer_events,
            prompt_version,
            original_passed
        FROM shadow_rewrite_log
        WHERE experiment_id = 'phase15.3.v1'
          AND status IN ('success', 'validation_failed')
          AND contract_data IS NOT NULL
          AND original_passed = false
          AND writer_events IS NOT NULL
          AND prompt_version = 'phase15.4c.contract_reinforced.v1'
    """)
    await conn.close()

    queried_samples = len(rows)
    print("=" * 80)
    print("Phase 15.5/15.6-B/D1: Field Semantic Diagnosis")
    print("=" * 80)
    print(f"查询到 Treatment 样本: {queried_samples}")

    if queried_samples != EXPECTED_TREATMENT_SAMPLES:
        print(f"⚠️ 警告: 预期 {EXPECTED_TREATMENT_SAMPLES} 个样本，实际查询到 {queried_samples} 个")
        print(f"   继续分析所有 {queried_samples} 个样本。")

    print()

    # 存储所有 requirement 级别的结果
    all_results = []  # 每个元素: (scene_id, req_status, detail)

    for row in rows:
        scene_id = row["scene_id"]
        try:
            contract_dict = json.loads(row["contract_data"]) if isinstance(row["contract_data"], str) else row["contract_data"]
            contract = PlanningContract(**contract_dict)
        except Exception:
            continue

        requirements = contract.observables.state_changes
        if not requirements:
            continue

        writer_events = row["writer_events"]
        if isinstance(writer_events, str):
            try:
                writer_events = json.loads(writer_events)
            except:
                continue
        if not isinstance(writer_events, list):
            continue

        used_indices = set()
        for sc in requirements:
            result, used_idx = match_requirement_to_events(sc, writer_events, used_indices)
            if used_idx is not None:
                used_indices.add(used_idx)
            all_results.append({
                "scene_id": scene_id,
                "req_type": sc.type,
                "status": result["status"],
                "reason": result["reason"],
                "matched_event": result["matched_event"],
            })

    # ============================================================
    # 统计
    # ============================================================
    status_counts = defaultdict(int)
    type_breakdown = defaultdict(lambda: defaultdict(int))

    for r in all_results:
        status_counts[r["status"]] += 1
        type_breakdown[r["req_type"]][r["status"]] += 1

    total_requirements = len(all_results)
    print(f"总 requirements: {total_requirements}")
    print()

    print("📊 总体状态分布:")
    print("-" * 40)
    for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / total_requirements * 100
        print(f"  {status}: {count} ({pct:.2f}%)")

    print()
    print("📊 按 type 细分:")
    print("-" * 40)
    for req_type, sub in sorted(type_breakdown.items()):
        print(f"  {req_type}:")
        for status, count in sorted(sub.items(), key=lambda x: x[1], reverse=True):
            pct = count / sum(sub.values()) * 100
            print(f"    {status}: {count} ({pct:.2f}%)")

    # ============================================================
    # 结论建议
    # ============================================================
    print()
    print("=" * 80)
    print("结论建议:")
    print("=" * 80)

    no_event_count = status_counts.get("NO_EVENT", 0)
    type_match_count = status_counts.get("TYPE_MATCH", 0)
    identity_match_count = status_counts.get("IDENTITY_MATCH", 0)

    print(f"🔍 IDENTITY_MATCH: {identity_match_count} ({identity_match_count/total_requirements*100:.1f}%)")
    print(f"🔍 TYPE_MATCH (但 identity 不同): {type_match_count} ({type_match_count/total_requirements*100:.1f}%)")
    print(f"🔍 NO_EVENT: {no_event_count} ({no_event_count/total_requirements*100:.1f}%)")

    if no_event_count > total_requirements * 0.3:
        print("\n⚠️ 高 NO_EVENT → Writer 仍然存在 requirement omission")
        print("   → 建议: 检查 Writer 是否未能感知到某些 Contract 要求")
    elif type_match_count > total_requirements * 0.3:
        print("\n⚠️ 高 TYPE_MATCH → Writer 产生了同类型事件，但 identity 字段不匹配")
        print("   → 建议: 检查 Writer 生成事件时使用的 identity 字段")
    elif identity_match_count > total_requirements * 0.7:
        print("\n✅ 高 IDENTITY_MATCH → Writer 正确执行了 Contract，问题可能在 Validator 的匹配逻辑")
        print("   → 建议: 检查 Validator 是否识别了 Writer 使用的 event type 别名")
    else:
        print("\n📊 混合型问题，建议结合具体样本分析")

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(stage_b_diagnose())