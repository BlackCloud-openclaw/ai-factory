#!/usr/bin/env python
"""
Phase 15.5-B: Field Semantic Diagnosis (Final Executable)

对 Treatment 样本进行字段级匹配分析。
分层诊断：TYPE → IDENTITY → FIELD

核心修正：
1. plot_flag identity: [("name",), ("flag",)] 真正的 alternatives
2. IDENTITY_MISMATCH 不消耗 candidates
3. FIELD 状态按优先级归类
4. 类型匹配采用精确匹配（基于 StateChangeType 枚举），无别名 fallback
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
from src.writing.state_change_types import StateChangeType

DSN = "postgresql://woami:kali@localhost:5432/ai_factory"

EXPECTED_TREATMENT_SAMPLES = 56

# ============================================================
# 类型匹配：使用枚举值，精确匹配
# ============================================================
VALID_TYPES = set(StateChangeType.values())

def get_allowed_event_types(req_type: str) -> set:
    # 只允许精确匹配，不进行别名转换
    if req_type not in VALID_TYPES:
        raise RuntimeError(f"Invalid StateChange.type: {req_type}")
    return {req_type}

# ============================================================
# IDENTITY_SCHEMA: 真正的 alternatives
# ============================================================
IDENTITY_SCHEMA = {
    "plot_flag": [
        ("name",),
        ("flag",),
    ],
    "knowledge_gain": [
        ("name",),
    ],
    "inventory_acquire": [
        ("actor", "item"),
    ],
    "location_change": [
        ("actor", "location"),
    ],
    "relationship_change": [
        ("from_char", "to_char"),
    ],
    "realm_change": [
        ("actor",),
    ],
}

# ============================================================
# FIELD_SCHEMA: 分离 identity 和 value
# ============================================================
FIELD_SCHEMA = {
    "plot_flag": {
        "identity_fields": [("name",), ("flag",)],
        "value_fields": ["value"],
    },
    "knowledge_gain": {
        "identity_fields": [("name",)],
        "value_fields": ["value"],
    },
    "inventory_acquire": {
        "identity_fields": [("actor", "item")],
        "value_fields": ["quantity"],
    },
    "location_change": {
        "identity_fields": [("actor", "location")],
        "value_fields": [],
    },
    "relationship_change": {
        "identity_fields": [("from_char", "to_char")],
        "value_fields": ["delta"],
    },
    "realm_change": {
        "identity_fields": [("actor",)],
        "value_fields": ["to_major_realm", "to_minor_stage"],
    },
}

# ============================================================
# 字段比较工具
# ============================================================
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
# identity 提取（无跨字段 fallback）
# ============================================================
def get_event_identity(event, fields_tuple: Tuple[str, ...]) -> Tuple:
    return tuple(
        normalize_value(event.get(f))
        for f in fields_tuple
    )

def get_requirement_identity(sc, fields_tuple: Tuple[str, ...]) -> Tuple:
    return tuple(normalize_value(getattr(sc, f, None)) for f in fields_tuple)

def identity_match(sc, event, identity_options: List[Tuple[str, ...]]) -> bool:
    for fields_tuple in identity_options:
        req_id = get_requirement_identity(sc, fields_tuple)
        evt_id = get_event_identity(event, fields_tuple)
        if all(v is not None for v in req_id) and req_id == evt_id:
            return True
    return False

# ============================================================
# FIELD 状态先收集、后按优先级归类
# ============================================================
def field_match(sc, event, field_schema: Dict) -> Dict:
    result = {
        "status": "FIELD_MATCH",
        "matched": [],
        "mismatched": [],
        "missing": [],
        "unexpected": [],
    }

    identity_options = field_schema.get("identity_fields", [])

    if not identity_match(sc, event, identity_options):
        result["status"] = "FIELD_MISMATCH"
        result["mismatched"].append({
            "field": "identity",
            "required": "any of " + str(identity_options),
            "actual": "none matched",
        })
        return result

    for field in field_schema.get("value_fields", []):
        req_val = getattr(sc, field, None)
        evt_val = event.get(field)

        if values_equal(req_val, evt_val):
            result["matched"].append(field)
        elif req_val is not None and evt_val is None:
            result["missing"].append(field)
        elif req_val is None and evt_val is not None:
            result["unexpected"].append({
                "field": field,
                "actual": evt_val,
            })
        else:
            result["mismatched"].append({
                "field": field,
                "required": req_val,
                "actual": evt_val,
            })

    # 按优先级统一决定状态
    if result["mismatched"]:
        result["status"] = "FIELD_MISMATCH"
    elif result["missing"]:
        result["status"] = "REQUIRED_FIELD_MISSING"
    elif result["unexpected"]:
        result["status"] = "UNEXPECTED_FIELD"
    else:
        result["status"] = "FIELD_MATCH"

    return result

# ============================================================
# 匹配引擎
# ============================================================
def match_requirements_to_events(
    requirements: List,
    events: List,
) -> Dict[str, Any]:
    available_events = list(events) if events else []
    used_event_indices = set()
    matched_requirement_indices = set()
    results = []

    # 第一轮：IDENTITY_MATCH
    for idx, sc in enumerate(requirements):
        req_type = sc.type
        allowed_types = get_allowed_event_types(req_type)
        identity_options = IDENTITY_SCHEMA.get(req_type, [])

        if not identity_options:
            continue

        matched_idx = None
        for i, event in enumerate(available_events):
            if i in used_event_indices:
                continue
            if event.get("type") not in allowed_types:
                continue
            if identity_match(sc, event, identity_options):
                matched_idx = i
                break

        if matched_idx is not None:
            used_event_indices.add(matched_idx)
            matched_requirement_indices.add(idx)
            results.append({
                "requirement_index": idx,
                "requirement": sc,
                "matched_event": available_events[matched_idx],
                "type_match": True,
                "identity_status": "IDENTITY_MATCH",
                "field_status": None,
                "field_result": None,
            })

    # 第二轮：IDENTITY_MISMATCH / TYPE_ONLY_CANDIDATE / NO_EVENT
    for idx, sc in enumerate(requirements):
        if idx in matched_requirement_indices:
            continue

        req_type = sc.type
        allowed_types = get_allowed_event_types(req_type)
        identity_options = IDENTITY_SCHEMA.get(req_type, [])

        candidate_indices = [
            i for i, event in enumerate(available_events)
            if i not in used_event_indices
            and event.get("type") in allowed_types
        ]

        if not candidate_indices:
            results.append({
                "requirement_index": idx,
                "requirement": sc,
                "matched_event": None,
                "type_match": False,
                "identity_status": "NO_EVENT",
                "field_status": "NOT_EVALUATED",
                "field_result": None,
            })
            continue

        if identity_options:
            identity_matched = False
            for cand_idx in candidate_indices:
                if identity_match(sc, available_events[cand_idx], identity_options):
                    identity_matched = True
                    matched_requirement_indices.add(idx)
                    used_event_indices.add(cand_idx)
                    results.append({
                        "requirement_index": idx,
                        "requirement": sc,
                        "matched_event": available_events[cand_idx],
                        "type_match": True,
                        "identity_status": "IDENTITY_MATCH",
                        "field_status": None,
                        "field_result": None,
                    })
                    break

            if not identity_matched:
                # IDENTITY_MISMATCH 不消耗 candidates
                results.append({
                    "requirement_index": idx,
                    "requirement": sc,
                    "matched_event": None,
                    "type_match": True,
                    "identity_status": "IDENTITY_MISMATCH",
                    "field_status": "NOT_EVALUATED",
                    "field_result": None,
                    "identity_candidates": [
                        available_events[i] for i in candidate_indices
                    ],
                })
        else:
            used_event_indices.add(candidate_indices[0])
            results.append({
                "requirement_index": idx,
                "requirement": sc,
                "matched_event": available_events[candidate_indices[0]],
                "type_match": True,
                "identity_status": "TYPE_ONLY_CANDIDATE",
                "field_status": "NOT_EVALUATED",
                "field_result": None,
            })

    # 第三轮：FIELD_COMPARE（仅对 IDENTITY_MATCH）
    for r in results:
        if r["identity_status"] != "IDENTITY_MATCH":
            continue

        sc = r["requirement"]
        event = r["matched_event"]
        req_type = sc.type
        field_schema = FIELD_SCHEMA.get(req_type, {})

        if not field_schema:
            r["field_status"] = "NOT_EVALUATED"
            continue

        field_result = field_match(sc, event, field_schema)
        r["field_result"] = field_result
        r["field_status"] = field_result["status"]

    # EXTRA_EVENTS: 未被使用的 Writer events
    extra_events = [
        event for i, event in enumerate(available_events)
        if i not in used_event_indices
    ]

    type_coverage = "FULL" if all(
        r["type_match"] for r in results
    ) else "PARTIAL"

    return {
        "requirement_results": results,
        "extra_events": extra_events,
        "total_requirements": len(requirements),
        "total_events": len(events),
        "type_coverage": type_coverage,
        "identity_match_count": sum(1 for r in results if r["identity_status"] == "IDENTITY_MATCH"),
        "identity_mismatch_count": sum(1 for r in results if r["identity_status"] == "IDENTITY_MISMATCH"),
        "no_event_count": sum(1 for r in results if r["identity_status"] == "NO_EVENT"),
        "type_only_count": sum(1 for r in results if r["identity_status"] == "TYPE_ONLY_CANDIDATE"),
    }


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
    print("Phase 15.5-B: Field Semantic Diagnosis (Final)")
    print("=" * 80)
    print(f"查询到 Treatment 样本: {queried_samples}")

    if queried_samples != EXPECTED_TREATMENT_SAMPLES:
        print(f"⚠️ 警告: 预期 {EXPECTED_TREATMENT_SAMPLES} 个样本，实际查询到 {queried_samples} 个")
        print(f"   请确认数据完整性后重试。")
        return

    print()

    # 分析样本
    analyzed_results = []
    skipped_count = 0
    skip_reasons = defaultdict(int)

    for row in rows:
        scene_id = row["scene_id"]

        try:
            contract_dict = json.loads(row["contract_data"]) if isinstance(row["contract_data"], str) else row["contract_data"]
            contract = PlanningContract(**contract_dict)
        except Exception as e:
            skipped_count += 1
            skip_reasons["contract_parse_failure"] += 1
            continue

        requirements = contract.observables.state_changes
        if not requirements:
            skipped_count += 1
            skip_reasons["no_requirements"] += 1
            continue

        writer_events = row["writer_events"]
        if isinstance(writer_events, str):
            try:
                writer_events = json.loads(writer_events)
            except:
                skipped_count += 1
                skip_reasons["writer_events_parse_failure"] += 1
                continue
        if not isinstance(writer_events, list):
            skipped_count += 1
            skip_reasons["writer_events_not_list"] += 1
            continue

        result = match_requirements_to_events(requirements, writer_events)
        analyzed_results.append({
            "scene_id": scene_id,
            "prompt_version": row["prompt_version"],
            "type_coverage": result["type_coverage"],
            "identity_match_count": result["identity_match_count"],
            "identity_mismatch_count": result["identity_mismatch_count"],
            "no_event_count": result["no_event_count"],
            "type_only_count": result["type_only_count"],
            "extra_events_count": len(result["extra_events"]),
            "raw_result": result,
        })

    analyzed_samples = len(analyzed_results)

    print(f"有效分析: {analyzed_samples}")
    print(f"跳过:     {skipped_count}")
    if skipped_count > 0:
        print("  跳过原因:")
        for reason, count in skip_reasons.items():
            print(f"    - {reason}: {count}")
    print()

    # 统计
    type_coverage_stats = {"FULL": 0, "PARTIAL": 0}
    identity_stats = defaultdict(int)
    field_stats = defaultdict(int)
    extra_event_counts = []
    total_requirements = 0
    total_events = 0

    for d in analyzed_results:
        result = d["raw_result"]
        type_coverage_stats[result["type_coverage"]] += 1
        for r in result["requirement_results"]:
            identity_stats[r["identity_status"]] += 1
            if r["field_status"]:
                field_stats[r["field_status"]] += 1
        extra_event_counts.append(len(result["extra_events"]))
        total_requirements += result["total_requirements"]
        total_events += result["total_events"]

    # 输出报告
    print("📊 TYPE_COVERAGE (场景级):")
    print("-" * 40)
    for status, count in type_coverage_stats.items():
        pct = count / analyzed_samples * 100 if analyzed_samples > 0 else 0
        print(f"  {status}: {count} ({pct:.2f}%)")

    print()
    print("📊 IDENTITY_STATUS (requirement级):")
    print("-" * 40)
    total_identity = sum(identity_stats.values())
    for status, count in sorted(identity_stats.items(), key=lambda x: x[1], reverse=True):
        pct = count / total_identity * 100 if total_identity > 0 else 0
        print(f"  {status}: {count} ({pct:.2f}%)")

    print()
    print("📊 FIELD_STATUS (仅 IDENTITY_MATCH):")
    print("-" * 40)
    total_field = sum(field_stats.values())
    if total_field > 0:
        for status, count in sorted(field_stats.items(), key=lambda x: x[1], reverse=True):
            pct = count / total_field * 100 if total_field > 0 else 0
            print(f"  {status}: {count} ({pct:.2f}%)")
    else:
        print("  (无 IDENTITY_MATCH 样本)")

    print()
    print("📊 EXTRA_EVENTS (场景级):")
    print("-" * 40)
    total_extra = sum(extra_event_counts)
    avg_extra = total_extra / len(extra_event_counts) if extra_event_counts else 0
    print(f"  总 extra_events: {total_extra}")
    print(f"  平均 per scene: {avg_extra:.2f}")
    print(f"  有 extra 的场景: {sum(1 for c in extra_event_counts if c > 0)}/{analyzed_samples}")

    print()
    print("📊 总体统计:")
    print("-" * 40)
    print(f"  总 requirements: {total_requirements}")
    print(f"  总 writer_events: {total_events}")

    # 结论建议
    print()
    print("=" * 80)
    print("结论建议:")
    print("=" * 80)

    identity_match_ratio = identity_stats.get("IDENTITY_MATCH", 0) / total_identity * 100 if total_identity else 0
    identity_mismatch_ratio = identity_stats.get("IDENTITY_MISMATCH", 0) / total_identity * 100 if total_identity else 0
    no_event_ratio = identity_stats.get("NO_EVENT", 0) / total_identity * 100 if total_identity else 0
    type_only_ratio = identity_stats.get("TYPE_ONLY_CANDIDATE", 0) / total_identity * 100 if total_identity else 0

    field_match_ratio = field_stats.get("FIELD_MATCH", 0) / total_field * 100 if total_field else 0
    field_mismatch_ratio = field_stats.get("FIELD_MISMATCH", 0) / total_field * 100 if total_field else 0
    field_missing_ratio = field_stats.get("REQUIRED_FIELD_MISSING", 0) / total_field * 100 if total_field else 0
    unexpected_ratio = field_stats.get("UNEXPECTED_FIELD", 0) / total_field * 100 if total_field else 0

    print(f"🔍 IDENTITY_MATCH: {identity_match_ratio:.1f}%")
    print(f"🔍 IDENTITY_MISMATCH: {identity_mismatch_ratio:.1f}%")
    print(f"🔍 NO_EVENT: {no_event_ratio:.1f}%")
    print(f"🔍 TYPE_ONLY_CANDIDATE: {type_only_ratio:.1f}%")
    print()
    print(f"🔍 FIELD_MATCH: {field_match_ratio:.1f}%")
    print(f"🔍 FIELD_MISMATCH: {field_mismatch_ratio:.1f}%")
    print(f"🔍 REQUIRED_FIELD_MISSING: {field_missing_ratio:.1f}%")
    print(f"🔍 UNEXPECTED_FIELD: {unexpected_ratio:.1f}%")

    print()
    print("归因判断:")
    if identity_match_ratio > 70 and field_match_ratio > 50:
        print("  ✅ Writer 执行质量良好，问题可能在 Validator 匹配逻辑")
        print("     → 建议: 检查 SemanticValidator 的匹配阈值和规则")
    elif identity_mismatch_ratio > 30:
        print("  ⚠️ 高 IDENTITY_MISMATCH → Writer 产生了同类型但错误目标的事件")
        print("     → 建议: 检查 Writer 如何确定事件的目标标识")
    elif field_mismatch_ratio > 30:
        print("  ⚠️ 高 FIELD_MISMATCH → Writer 知道目标但字段值错误")
        print("     → 建议: 检查 Writer 生成具体字段值的能力")
    elif field_missing_ratio > 30:
        print("  ⚠️ 高 REQUIRED_FIELD_MISSING → Writer 遗漏了必要字段")
        print("     → 建议: 检查 Writer 是否完整生成 Contract 要求的字段")
    elif no_event_ratio > 30:
        print("  ⚠️ 高 NO_EVENT → Writer 仍然存在 requirement omission")
        print("     → 建议: 检查 Writer 是否未能感知到某些 Contract 要求")
    else:
        print("  📊 混合型问题，建议结合具体样本分析")

    if total_extra > 0:
        print()
        print(f"⚠️ 存在 {total_extra} 个 EXTRA_EVENTS (平均 {avg_extra:.2f}/场景)")
        print("   → 建议: 检查 Writer 是否过度生成事件")

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(stage_b_diagnose())