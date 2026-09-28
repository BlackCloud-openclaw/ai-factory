#!/usr/bin/env python
"""
Phase 15.5-A: Deterministic Event Attempt Analysis (Frozen Version)

职责：
- 不调用 SemanticValidator
- 只做机械事件类型匹配
- 区分 event_parse_failure 与合法空 events
- 按 requirement 逐个匹配，避免 writer_types 集合导致的虚高计数

输出：
- NO_ATTEMPT
- PARTIAL_TYPE_ATTEMPT
- FULL_TYPE_ATTEMPT
- EVENT_PARSE_FAILURE
- CONTRACT_PARSE_FAILURE
- NO_REQUIREMENTS
"""

import sys
import asyncio
import asyncpg
import json
import re
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.writing.planning_contract import PlanningContract

DSN = "postgresql://woami:kali@localhost:5432/ai_factory"

# 类型别名映射：Contract state_change.type → Writer event.type 允许集
EVENT_TYPE_ALIASES = {
    "realm_change": {"realm_change", "realm_upgrade"},
    "plot_flag": {"plot_flag", "plot_flag_set"},
    "inventory_acquire": {"item_acquire", "inventory_acquire"},
    "location_change": {"location_change", "location_enter"},
    "relationship_change": {"relationship_change"},
    "knowledge_gain": {"knowledge_gain"},
}


def extract_writer_events(text: str):
    """
    从 Writer 输出中提取 events 数组。
    返回: (events, parse_ok)
    """
    if not text:
        return [], False

    # 检查是否包含 "events": 字段
    if '"events"' not in text:
        return [], False

    # 寻找 events 数组的起始位置
    start = text.find('"events"')
    if start == -1:
        return [], False

    colon = text.find(":", start)
    if colon == -1:
        return [], False

    # 从冒号后找第一个 [
    bracket = text.find("[", colon)
    if bracket == -1:
        return [], False

    try:
        decoder = json.JSONDecoder()
        data, end = decoder.raw_decode(text[bracket:])
    except json.JSONDecodeError:
        return [], False

    if not isinstance(data, list):
        return [], False

    return data, True


def analyze_attempts(contract, writer_events):
    """
    对每个 requirement，尝试匹配一个 writer event。
    一个 writer event 只能匹配一个 requirement。
    返回: {
        attempted_count: int,
        total_count: int,
        missing_types: list,
        matched: list,
    }
    """
    requirements = contract.observables.state_changes

    if not requirements:
        return {
            "attempted_count": 0,
            "total_count": 0,
            "missing_types": [],
            "matched": [],
        }

    # 复制可用事件列表，按顺序消耗
    available_events = list(writer_events)
    matched = []
    missing_types = []

    for idx, sc in enumerate(requirements):
        allowed_types = EVENT_TYPE_ALIASES.get(sc.type, {sc.type})

        event_index = next(
            (
                i for i, event in enumerate(available_events)
                if event.get("type") in allowed_types
            ),
            None,
        )

        if event_index is None:
            missing_types.append(sc.type)
        else:
            event = available_events.pop(event_index)
            matched.append({
                "requirement_index": idx,
                "required_type": sc.type,
                "matched_event_type": event.get("type"),
            })

    return {
        "attempted_count": len(matched),
        "total_count": len(requirements),
        "missing_types": missing_types,
        "matched": matched,
    }


async def stage_a_diagnose():
    conn = await asyncpg.connect(DSN)

    # 统计总记录数（用于分母）
    total_records = await conn.fetchval("""
        SELECT COUNT(*) 
        FROM shadow_rewrite_log
        WHERE experiment_id = 'phase15.3.v1'
          AND status IN ('success', 'validation_failed')
          AND contract_data IS NOT NULL
    """)

    original_pass = await conn.fetchval("""
        SELECT COUNT(*) 
        FROM shadow_rewrite_log
        WHERE experiment_id = 'phase15.3.v1'
          AND status IN ('success', 'validation_failed')
          AND contract_data IS NOT NULL
          AND original_passed = true
    """)

    # 获取所有 FAIL 样本
    rows = await conn.fetch("""
        SELECT
            scene_id,
            contract_data,
            original_text,
            prompt_version,
            original_passed
        FROM shadow_rewrite_log
        WHERE experiment_id = 'phase15.3.v1'
          AND status IN ('success', 'validation_failed')
          AND contract_data IS NOT NULL
          AND original_passed = false
    """)
    await conn.close()

    print("=" * 80)
    print("Phase 15.5-A: Deterministic Event Attempt Analysis")
    print("=" * 80)
    print(f"Total experiment records: {total_records}")
    print(f"Original PASS:            {original_pass}")
    print(f"Original FAIL:            {len(rows)}")
    print()

    categories = defaultdict(int)
    detail_by_prompt = defaultdict(lambda: defaultdict(int))

    for row in rows:
        scene_id = row["scene_id"]

        # Contract 解析
        try:
            contract_dict = json.loads(row["contract_data"]) if isinstance(row["contract_data"], str) else row["contract_data"]
            contract = PlanningContract(**contract_dict)
        except Exception:
            categories["contract_parse_failure"] += 1
            detail_by_prompt[row["prompt_version"]]["contract_parse_failure"] += 1
            continue

        # 无 requirements
        if not contract.observables.state_changes:
            categories["no_requirements"] += 1
            detail_by_prompt[row["prompt_version"]]["no_requirements"] += 1
            continue

        # 提取 Writer events
        writer_events, parse_ok = extract_writer_events(row["original_text"])

        if not parse_ok:
            categories["event_parse_failure"] += 1
            detail_by_prompt[row["prompt_version"]]["event_parse_failure"] += 1
            continue

        # 分析 Attempt
        result = analyze_attempts(contract, writer_events)
        attempted = result["attempted_count"]
        total = result["total_count"]

        if attempted == 0:
            cat = "no_attempt"
        elif attempted == total:
            cat = "full_type_attempt"
        else:
            cat = "partial_type_attempt"

        categories[cat] += 1
        detail_by_prompt[row["prompt_version"]][cat] += 1

    total_diagnosed = sum(categories.values())
    print("📊 Stage A 分类结果:")
    print("-" * 40)
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count} ({count/total_diagnosed*100:.2f}%)")
    print()

    print("📊 按实验分组对比:")
    print("-" * 40)
    print(f"  {'Category':<22} {'Baseline':<12} {'Treatment':<12}")
    print(f"  {'-'*22:<22} {'-'*12:<12} {'-'*12:<12}")
    for cat in sorted(set().union(*[set(d.keys()) for d in detail_by_prompt.values()])):
        b = detail_by_prompt.get("phase15.3.v1", {}).get(cat, 0)
        t = detail_by_prompt.get("phase15.4c.contract_reinforced.v1", {}).get(cat, 0)
        print(f"  {cat:<22} {b:<12} {t:<12}")

    print("\n" + "=" * 80)
    print("Stage A 诊断完成。")
    print("下一步：如果 FULL_TYPE_ATTEMPT 占比 > 20%，进入 Stage B (字段语义诊断)。")
    print("如果 NO_ATTEMPT 仍占绝大多数，15.6 应聚焦于 Writer 执行机制。")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(stage_a_diagnose())