#!/usr/bin/env python
"""
C3.3b: Validator V2 LLM Evaluation

使用真实 SemanticJudge 对 ValidationV2 进行语义评估。
不设硬门槛，仅收集通过率与案例级失败数据。

类别:
  1. realm_change   —— 结构不变量（含"LLM 不得提升 SUPPORTED"）
  2. knowledge_gain —— 理解 vs 尝试 vs 听闻
  3. plot_flag      —— 三值语义边界：SUPPORTED / CONTRADICTED / INSUFFICIENT

输出:
  - stdout: 可读摘要
  - reports/c3_3b_llm_eval_{ts}.json: 详细结果

C3.4.1 修订:
  - plot_02 从 "not_occurred / INSUFFICIENT" 改为 "contradicted / CONTRADICTED"
  - 新增 plot_03_silent（期望 INSUFFICIENT），补齐三值覆盖

C3.4.2 修订:
  - run_case 直接展开 result.to_dict()，不再手写字段白名单
  - 这样 retrieved_evidence_count / retrieved_evidence_ids 等
    观测字段会自动出现在 JSON 报告中
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.writing.planning_contract import StateChange
from src.writing.validation_v2.claim_builder import ClaimBuilder
from src.writing.validation_v2.validator import ValidationV2
from src.writing.validation_v2.semantic_judge import SemanticJudge
from src.writing.validation_v2.models import Verdict, MatchLayer


# ============================================================
# 测试用例
# ============================================================

REALM_CASES: List[Dict[str, Any]] = [
    {
        "name": "realm_01_feature_description",
        "expectation": "INSUFFICIENT",
        "rationale": "特征描述（威压/气息）不构成事实完成",
        "state_change": {
            "id": "r01",
            "type": "realm_change",
            "actor": "林逸",
            "to_major_realm": "大乘",
            "to_minor_stage": 1,
        },
        "writer_events": [],
        "scene_text": "林逸身上浮现出大乘期修士才有的威压，令众人心惊。",
    },
    {
        "name": "realm_02_attempting",
        "expectation": "INSUFFICIENT",
        "rationale": "尝试突破中，尚未完成",
        "state_change": {
            "id": "r02",
            "type": "realm_change",
            "actor": "林逸",
            "to_major_realm": "大乘",
            "to_minor_stage": 1,
        },
        "writer_events": [],
        "scene_text": "林逸盘膝而坐，试图冲击大乘境，经脉中灵气翻涌。",
    },
    {
        "name": "realm_03_completed_with_structural_match",
        "expectation": "SUPPORTED",
        "rationale": "完成突破 + writer_events 结构精确匹配 → EXACT 短路",
        "state_change": {
            "id": "r03",
            "type": "realm_change",
            "actor": "林逸",
            "to_major_realm": "大乘",
            "to_minor_stage": 1,
        },
        "writer_events": [
            {
                "type": "realm_upgrade",
                "actor": "林逸",
                "to_major_realm": "大乘",
                "to_minor_stage": 1,
            }
        ],
        "scene_text": "他气息暴涨，终于踏入大乘境。",
    },
    {
        "name": "realm_04_completed_no_structural_match",
        "expectation": "INSUFFICIENT",
        "rationale": "★ 关键不变量：LLM 判 SUPPORTED 但无结构匹配 → 降级 INSUFFICIENT",
        "state_change": {
            "id": "r04",
            "type": "realm_change",
            "actor": "林逸",
            "to_major_realm": "大乘",
            "to_minor_stage": 1,
        },
        "writer_events": [],
        "scene_text": "他终于踏入大乘境，周身灵光流转。",
    },
    {
        "name": "realm_05_explicit_failure",
        "expectation": "CONTRADICTED",
        "rationale": "明确失败 → LLM 应判 CONTRADICTED",
        "state_change": {
            "id": "r05",
            "type": "realm_change",
            "actor": "林逸",
            "to_major_realm": "大乘",
            "to_minor_stage": 1,
        },
        "writer_events": [],
        "scene_text": "冲击失败，林逸脸色苍白，依旧停留在金丹九层。",
    },
    {
        "name": "realm_06_stage_mismatch",
        "expectation": "INSUFFICIENT",
        "rationale": "结构匹配但 stage 不匹配 → 直判 INSUFFICIENT（不调 LLM）",
        "state_change": {
            "id": "r06",
            "type": "realm_change",
            "actor": "林逸",
            "to_major_realm": "大乘",
            "to_minor_stage": 1,
        },
        "writer_events": [
            {
                "type": "realm_upgrade",
                "actor": "林逸",
                "to_major_realm": "大乘",
                "to_minor_stage": 2,
            }
        ],
        "scene_text": "",
    },
]


KNOWLEDGE_CASES: List[Dict[str, Any]] = [
    {
        "name": "knowledge_01_understood",
        "expectation": "SUPPORTED",
        "rationale": "已完成的理解",
        "state_change": {
            "id": "k01",
            "type": "knowledge_gain",
            "actor": "林逸",
            "name": "阵法原理",
        },
        "writer_events": [],
        "scene_text": "林逸终于参透了阵法原理，心中豁然开朗。",
    },
    {
        "name": "knowledge_02_attempting",
        "expectation": "INSUFFICIENT",
        "rationale": "尝试理解中，未完成",
        "state_change": {
            "id": "k02",
            "type": "knowledge_gain",
            "actor": "林逸",
            "name": "阵法原理",
        },
        "writer_events": [],
        "scene_text": "林逸试图理解阵法原理，但始终不得要领。",
    },
    {
        "name": "knowledge_03_heard_only",
        "expectation": "INSUFFICIENT",
        "rationale": "仅听闻，未内化",
        "state_change": {
            "id": "k03",
            "type": "knowledge_gain",
            "actor": "林逸",
            "name": "阵法原理",
        },
        "writer_events": [],
        "scene_text": "林逸听闻过阵法原理，却从未真正研习。",
    },
]


# ============================================================
# plot_flag：三值覆盖（C3.4.1 修订）
# ============================================================
#
# 三值语义边界:
#   SUPPORTED     证据表明事实已发生
#   CONTRADICTED  证据明确表明事实未发生 / 与命题冲突
#   INSUFFICIENT  场景未提供足够信息
# ============================================================

PLOT_FLAG_CASES: List[Dict[str, Any]] = [
    {
        "name": "plot_01_occurred",
        "expectation": "SUPPORTED",
        "rationale": "结果性证据：封印亮起、符文流转，事实已发生",
        "state_change": {
            "id": "p01",
            "type": "plot_flag",
            "name": "上古封印触发",
            "value": True,
        },
        "writer_events": [],
        "scene_text": "石壁上的封印骤然亮起，古老符文流转，封印被触发了。",
    },
    {
        "name": "plot_02_contradicted",
        "expectation": "CONTRADICTED",
        "rationale": "反证：封印处于未触发状态（沉睡、毫无动静），与命题冲突",
        "state_change": {
            "id": "p02",
            "type": "plot_flag",
            "name": "上古封印触发",
            "value": True,
        },
        "writer_events": [],
        "scene_text": "林逸打量石壁上的封印，它依旧沉睡着，毫无动静。",
    },
    {
        "name": "plot_03_silent",
        "expectation": "INSUFFICIENT",
        "rationale": "无证据：场景未提及封印状态，既未支持也未否定",
        "state_change": {
            "id": "p03",
            "type": "plot_flag",
            "name": "上古封印触发",
            "value": True,
        },
        "writer_events": [],
        "scene_text": "林逸在石壁前驻足片刻，随后转身离去。",
    },
]


ALL_CASES = [
    ("realm_change", REALM_CASES),
    ("knowledge_gain", KNOWLEDGE_CASES),
    ("plot_flag", PLOT_FLAG_CASES),
]


# ============================================================
# 运行器
# ============================================================

async def run_case(
    category: str,
    case: Dict[str, Any],
    validator: ValidationV2,
) -> Dict[str, Any]:
    sc = StateChange(**case["state_change"])
    claim = ClaimBuilder.from_state_change(sc, contract_id="c3_3b")

    t0 = time.time()
    try:
        result = await validator.validate_claim(
            claim=claim,
            writer_events=case["writer_events"],
            scene_text=case["scene_text"],
        )
        elapsed = time.time() - t0
        expected = case["expectation"]
        actual = result.verdict.value

        # C3.4.2: 直接展开 result.to_dict()，避免手写字段白名单
        # 这样新增观测字段（如 retrieved_evidence_count / retrieved_evidence_ids）
        # 会自动出现在 JSON 报告中。
        result_dict = result.to_dict()

        return {
            # ---- 元信息 ----
            "name": case["name"],
            "category": category,
            "expectation": expected,
            "actual": actual,
            "passed": (actual == expected),
            "rationale": case["rationale"],
            # ---- 展开所有观测/判定字段 ----
            # 包含: verdict, matched_layer, confidence, reason, evidence,
            #       judgement, evidence_candidates_found, evidence_status,
            #       structural_check, fallback_applied, llm_invoked,
            #       retrieved_evidence_count, retrieved_evidence_ids
            **result_dict,
            # ---- 上下文 ----
            "scene_text": case["scene_text"][:120],
            "writer_events_count": len(case["writer_events"]),
            "elapsed_s": round(elapsed, 2),
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "name": case["name"],
            "category": category,
            "expectation": case["expectation"],
            "actual": "ERROR",
            "passed": False,
            "rationale": case["rationale"],
            "error": f"{type(e).__name__}: {e}",
            "elapsed_s": round(elapsed, 2),
        }


async def main() -> int:
    print("=" * 64)
    print("C3.3b: Validator V2 LLM Evaluation")
    print("=" * 64)
    print()

    judge = SemanticJudge()
    validator = ValidationV2(judge=judge)

    print(f"SemanticJudge model: {judge.model}")
    print(f"Timeout: {judge.timeout}s")
    print()

    all_results: List[Dict[str, Any]] = []

    for category, cases in ALL_CASES:
        print(f"### {category}  ({len(cases)} cases)")
        print("-" * 64)

        for case in cases:
            result = await run_case(category, case, validator)
            all_results.append(result)

            icon = "✅" if result["passed"] else "❌"
            print(f"  {icon} {result['name']}")
            print(
                f"      expected={result['expectation']:14s} "
                f"actual={result['actual']:14s} "
                f"conf={result.get('confidence', 0):.2f} "
                f"layer={result.get('matched_layer', 'N/A')}"
            )
            if not result["passed"]:
                reason = result.get("reason", result.get("error", ""))
                print(f"      ⚠️  reason: {reason[:100]}")
            print()

    # ---------- 汇总 ----------
    print("=" * 64)
    print("汇总")
    print("=" * 64)

    by_cat: Dict[str, Dict[str, int]] = {}
    for r in all_results:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = {"total": 0, "passed": 0}
        by_cat[cat]["total"] += 1
        if r["passed"]:
            by_cat[cat]["passed"] += 1

    total = len(all_results)
    total_passed = sum(1 for r in all_results if r["passed"])

    for cat, stats in by_cat.items():
        print(f"  {cat:20s}: {stats['passed']}/{stats['total']}")
    pct = (total_passed / total * 100) if total else 0.0
    print(f"  {'TOTAL':20s}: {total_passed}/{total} ({pct:.1f}%)")

    # ---------- 关键不变量单独校验 ----------
    print()
    print("=" * 64)
    print("关键不变量（realm_change）")
    print("=" * 64)

    key_invariants = {
        "realm_03_completed_with_structural_match": "EXACT 短路（无 LLM）",
        "realm_04_completed_no_structural_match":  "LLM 不得提升 SUPPORTED",
        "realm_05_explicit_failure":               "LLM 可以判 CONTRADICTED",
        "realm_06_stage_mismatch":                 "结构 MISMATCH 直判（无 LLM）",
    }
    for name, desc in key_invariants.items():
        r = next((x for x in all_results if x["name"] == name), None)
        if r is None:
            continue
        icon = "✅" if r["passed"] else "❌"
        print(f"  {icon} {desc}")
        print(f"      {name}: expected={r['expectation']}  actual={r['actual']}")

    # ---------- plot_flag 三值覆盖校验 ----------
    print()
    print("=" * 64)
    print("plot_flag 三值覆盖")
    print("=" * 64)
    for want in ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT"):
        matching = [r for r in all_results
                    if r["category"] == "plot_flag" and r["expectation"] == want]
        for r in matching:
            icon = "✅" if r["passed"] else "❌"
            print(f"  {icon} {r['name']:32s} expected={want}  actual={r['actual']}")

    # ---------- 保存详细结果 ----------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"c3_3b_llm_eval_{ts}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": ts,
                "total": total,
                "passed": total_passed,
                "pass_rate": round(pct, 2),
                "by_category": by_cat,
                "results": all_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"详细结果: {out_path}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))