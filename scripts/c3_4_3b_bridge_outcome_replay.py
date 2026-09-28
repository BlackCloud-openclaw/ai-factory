#!/usr/bin/env python
"""
C3.4.3B.5: Bridge Outcome Replay Analyzer

只读 bridge_outcome_audit + validator_v2_audit。
不修改任何生产逻辑。

边界（冻结）:
- 不计算 False Rescue
- 不计算 Rescue Precision
- 不基于 representative_claim_type 判断"失败 claim type"
- 所有输出为观察结果，非质量结论

C3.4.3B.5.v2 修订（路径 B 保守版）:
- 取消 claim_level_join_*：同一 (novel_id, volume, chapter, scene_idx)
  可以被 try_rescue 调用多次，且结果可能不同，
  该键不是一对一关联键。
- 改为分层展示：
    bridge_level_*   仅来自 bridge_outcome_audit
    claim_level_*    仅来自 validator_v2_audit
- 两者之间不做 JOIN，不做顺序配对。
- 若下游需要交叉参照，使用 (novel_id, volume_num, chapter_num, scene_idx)
  作为 scene-level association key，并明确标注：
      "scene-level association, not attempt-level lineage"

输入:
- bridge_outcome_audit  (bridge-level)
- validator_v2_audit    (claim-level)

输出:
- stdout: 人类可读摘要
- reports/c3_4_3b_bridge_outcome_replay_{ts}.json

用法:
  python scripts/c3_4_3b_bridge_outcome_replay.py
  python scripts/c3_4_3b_bridge_outcome_replay.py --novel_id simple_long_novel_001
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import asyncpg


SCHEMA_VERSION = "c3_4_3b_5.v2"


# ============================================================
# 数据库连接（独立 pool）
# ============================================================

async def _create_pool() -> asyncpg.Pool:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    database = os.getenv("POSTGRES_DB", "ai_factory")
    user = os.getenv("POSTGRES_USER", "woami")
    password = os.getenv("POSTGRES_PASSWORD", "kali")

    print(f"连接数据库: {user}@{host}:{port}/{database}")
    return await asyncpg.create_pool(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        min_size=1,
        max_size=2,
        timeout=10.0,
    )


# ============================================================
# 工具
# ============================================================

def _jsonb_to_list(v):
    """asyncpg 把 JSONB 返回为 str；兼容 str / list / None。"""
    if v is None:
        return None
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return None
    return v


# ============================================================
# 分析查询
# ============================================================

async def _build_where(novel_id: Optional[str]) -> tuple[str, list]:
    if novel_id:
        return "WHERE novel_id = $1", [novel_id]
    return "", []


async def _analyze(conn, where: str, args: list) -> dict:
    stats: dict = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(),
        "filters": {"novel_id": args[0] if args else None},
        "semantics_note": (
            "bridge_level_* and claim_level_* are independent. "
            "No attempt-level lineage is claimed. "
            "Cross-reference key (novel_id, volume_num, chapter_num, scene_idx) "
            "is scene-level association only."
        ),
    }

    # ------------------------------------------------------------
    # 0. 总量
    # ------------------------------------------------------------
    row = await conn.fetchrow(
        f"SELECT COUNT(*) AS c FROM bridge_outcome_audit {where}", *args
    )
    stats["total_rows"] = row["c"]
    if stats["total_rows"] == 0:
        return stats

    # ------------------------------------------------------------
    # 1. 按 chapter 的 Rescue Rate 时序
    # ------------------------------------------------------------
    rows = await conn.fetch(f"""
        SELECT
            novel_id,
            volume_num,
            chapter_num,
            COUNT(*) AS triggered,
            COUNT(*) FILTER (WHERE execution_status = 'completed') AS completed,
            COUNT(*) FILTER (WHERE rescued = TRUE) AS rescued,
            COUNT(DISTINCT scene_idx) AS scene_count
        FROM bridge_outcome_audit
        {where}
        {'AND' if where else 'WHERE'} trigger_status = 'triggered'
        GROUP BY novel_id, volume_num, chapter_num
        ORDER BY novel_id, volume_num, chapter_num
    """, *args)
    stats["by_chapter"] = [
        {
            "novel_id": r["novel_id"],
            "volume_num": r["volume_num"],
            "chapter_num": r["chapter_num"],
            "triggered": r["triggered"],
            "completed": r["completed"],
            "rescued": r["rescued"],
            "scene_count": r["scene_count"],
            "rescue_rate": round(
                (r["rescued"] / r["completed"]) if r["completed"] > 0 else 0.0,
                4,
            ),
        }
        for r in rows
    ]

    # ------------------------------------------------------------
    # 2. min_confidence 分布（仅 triggered + completed）
    # ------------------------------------------------------------
    row = await conn.fetchrow(f"""
        SELECT
            COUNT(*) AS n,
            MIN(min_confidence) AS min_v,
            percentile_cont(0.25) WITHIN GROUP (ORDER BY min_confidence) AS p25,
            percentile_cont(0.50) WITHIN GROUP (ORDER BY min_confidence) AS median,
            percentile_cont(0.75) WITHIN GROUP (ORDER BY min_confidence) AS p75,
            MAX(min_confidence) AS max_v
        FROM bridge_outcome_audit
        {where}
        {'AND' if where else 'WHERE'} trigger_status = 'triggered'
          AND execution_status = 'completed'
          AND min_confidence IS NOT NULL
    """, *args)
    stats["confidence_distribution"] = {
        "count": row["n"],
        "min": round(row["min_v"], 4) if row["min_v"] is not None else None,
        "p25": round(row["p25"], 4) if row["p25"] is not None else None,
        "median": round(row["median"], 4) if row["median"] is not None else None,
        "p75": round(row["p75"], 4) if row["p75"] is not None else None,
        "max": round(row["max_v"], 4) if row["max_v"] is not None else None,
    }

    # ------------------------------------------------------------
    # 3. 三变量交叉表
    # ------------------------------------------------------------
    rows = await conn.fetch(f"""
        SELECT all_supported, confidence_ok, rescued, COUNT(*) AS c
        FROM bridge_outcome_audit
        {where}
        {'AND' if where else 'WHERE'} trigger_status = 'triggered'
          AND execution_status = 'completed'
        GROUP BY all_supported, confidence_ok, rescued
        ORDER BY c DESC
    """, *args)
    stats["decision_matrix"] = [
        {
            "all_supported": r["all_supported"],
            "confidence_ok": r["confidence_ok"],
            "rescued": r["rescued"],
            "count": r["c"],
        }
        for r in rows
    ]

    # ------------------------------------------------------------
    # 4. trigger_status 分布
    # ------------------------------------------------------------
    rows = await conn.fetch(f"""
        SELECT trigger_status, COUNT(*) AS c
        FROM bridge_outcome_audit {where}
        GROUP BY trigger_status
    """, *args)
    raw = {r["trigger_status"]: r["c"] for r in rows}
    stats["trigger_status"] = {
        "original_passed": raw.get("original_passed", 0),
        "no_state_changes": raw.get("no_state_changes", 0),
        "no_production_types": raw.get("no_production_types", 0),
        "triggered": raw.get("triggered", 0),
    }

    # ------------------------------------------------------------
    # 5. execution_status 分布
    # ------------------------------------------------------------
    rows = await conn.fetch(f"""
        SELECT execution_status, COUNT(*) AS c
        FROM bridge_outcome_audit {where}
        GROUP BY execution_status
    """, *args)
    raw = {r["execution_status"]: r["c"] for r in rows}
    completed = raw.get("completed", 0)
    error = raw.get("error", 0)
    total = completed + error
    stats["execution_status"] = {
        "completed": completed,
        "error": error,
        "error_rate": round(error / total, 4) if total > 0 else 0.0,
    }

    # ------------------------------------------------------------
    # 6. representative_claim_type 分布
    # ------------------------------------------------------------
    rows = await conn.fetch(f"""
        SELECT
            representative_claim_type,
            COUNT(*) AS c,
            COUNT(*) FILTER (WHERE rescued = TRUE) AS rescued_count,
            COUNT(*) FILTER (WHERE rescued IS NOT TRUE) AS not_rescued_count
        FROM bridge_outcome_audit
        {where}
        {'AND' if where else 'WHERE'} trigger_status = 'triggered'
          AND execution_status = 'completed'
          AND representative_claim_type IS NOT NULL
        GROUP BY representative_claim_type
        ORDER BY c DESC
    """, *args)
    stats["representative_claim_types"] = [
        {
            "type": r["representative_claim_type"],
            "count": r["c"],
            "rescued": r["rescued_count"],
            "not_rescued": r["not_rescued_count"],
            "note": "representative, NOT failure claim type",
        }
        for r in rows
    ]

    # ------------------------------------------------------------
    # 7. Bridge-level 明细（纯 bridge 层数据，无 JOIN）
    # ------------------------------------------------------------
    rows = await conn.fetch(f"""
        SELECT
            id, novel_id, volume_num, chapter_num, scene_idx, scene_id,
            triggered, trigger_status, execution_status,
            rescued, all_supported, confidence_ok, min_confidence,
            representative_claim_id, representative_claim_type,
            representative_reason, representative_evidence_ids
        FROM bridge_outcome_audit
        {where}
        ORDER BY id
    """, *args)

    bridge_all: List[Dict[str, Any]] = []
    bridge_failed: List[Dict[str, Any]] = []

    for r in rows:
        entry = {
            "bridge_id": r["id"],
            "novel_id": r["novel_id"],
            "volume_num": r["volume_num"],
            "chapter_num": r["chapter_num"],
            "scene_idx": r["scene_idx"],
            "scene_id": r["scene_id"],
            "trigger_status": r["trigger_status"],
            "triggered": r["triggered"],
            "execution_status": r["execution_status"],
            "rescued": r["rescued"],
            "all_supported": r["all_supported"],
            "confidence_ok": r["confidence_ok"],
            "min_confidence": (
                round(r["min_confidence"], 4)
                if r["min_confidence"] is not None
                else None
            ),
            "representative_claim_id": r["representative_claim_id"],
            "representative_claim_type": r["representative_claim_type"],
            "representative_reason": r["representative_reason"],
            "representative_evidence_ids": _jsonb_to_list(
                r["representative_evidence_ids"]
            ),
        }
        bridge_all.append(entry)

        # rescue 失败的定义：triggered + completed + rescued != TRUE
        if (
            entry["trigger_status"] == "triggered"
            and entry["execution_status"] == "completed"
            and entry["rescued"] is not True
        ):
            bridge_failed.append(entry)

    stats["bridge_level_all"] = bridge_all
    stats["bridge_level_rescued_false"] = bridge_failed

    # ------------------------------------------------------------
    # 8. Claim-level 明细（纯 claim 层数据，无 JOIN）
    # ------------------------------------------------------------
    rows = await conn.fetch(f"""
        SELECT
            id, novel_id, volume_num, chapter_num, scene_idx, scene_id,
            mode, claim_id, state_change_type,
            structural_check,
            evidence_candidates_found,
            retrieved_evidence_count, retrieved_evidence_ids,
            llm_invoked, llm_raw_verdict, llm_raw_confidence, llm_raw_reason,
            llm_evidence_ids,
            final_verdict, matched_layer, final_confidence,
            fallback_applied, final_reason
        FROM validator_v2_audit
        {where}
        ORDER BY id
    """, *args)

    claim_all: List[Dict[str, Any]] = []
    for r in rows:
        claim_all.append({
            "audit_id": r["id"],
            "novel_id": r["novel_id"],
            "volume_num": r["volume_num"],
            "chapter_num": r["chapter_num"],
            "scene_idx": r["scene_idx"],
            "scene_id": r["scene_id"],
            "mode": r["mode"],
            "claim_id": r["claim_id"],
            "state_change_type": r["state_change_type"],
            "structural_check": r["structural_check"],
            "evidence_candidates_found": r["evidence_candidates_found"],
            "retrieved_evidence_count": r["retrieved_evidence_count"],
            "retrieved_evidence_ids": _jsonb_to_list(r["retrieved_evidence_ids"]),
            "llm_invoked": r["llm_invoked"],
            "llm_raw_verdict": r["llm_raw_verdict"],
            "llm_raw_confidence": (
                round(r["llm_raw_confidence"], 4)
                if r["llm_raw_confidence"] is not None
                else None
            ),
            "llm_raw_reason": r["llm_raw_reason"],
            "llm_evidence_ids": _jsonb_to_list(r["llm_evidence_ids"]),
            "final_verdict": r["final_verdict"],
            "matched_layer": r["matched_layer"],
            "final_confidence": (
                round(r["final_confidence"], 4)
                if r["final_confidence"] is not None
                else None
            ),
            "fallback_applied": r["fallback_applied"],
            "final_reason": r["final_reason"],
        })

    stats["claim_level_all"] = claim_all
    stats["claim_level_count"] = len(claim_all)

    return stats


# ============================================================
# 输出
# ============================================================

def _print_report(stats: dict) -> None:
    total = stats["total_rows"]

    print()
    print("=" * 72)
    print("C3.4.3B.5: Bridge Outcome Replay")
    print("=" * 72)
    print(f"Schema     : {stats['schema_version']}")
    print(f"Generated  : {stats['generated_at']}")
    print(f"Filters    : {stats['filters']}")
    print(f"Total rows : {total}")
    print()

    if total == 0:
        print("⚠️  无数据。")
        return

    # 1. by_chapter
    print("### 1. Rescue Rate 时序（按 chapter）")
    print(f"  {'novel':24s} {'vol':>3s} {'ch':>4s} "
          f"{'scenes':>6s} {'trig':>5s} {'comp':>5s} {'resc':>5s} {'rate':>7s}")
    print(f"  {'-'*24} {'-'*3} {'-'*4} {'-'*6} {'-'*5} {'-'*5} {'-'*5} {'-'*7}")
    for r in stats["by_chapter"]:
        print(
            f"  {r['novel_id']:24s} "
            f"{r['volume_num']:>3d} "
            f"{r['chapter_num']:>4d} "
            f"{r['scene_count']:>6d} "
            f"{r['triggered']:>5d} "
            f"{r['completed']:>5d} "
            f"{r['rescued']:>5d} "
            f"{r['rescue_rate']:>7.4f}"
        )
    print()

    # 2. confidence
    cd = stats["confidence_distribution"]
    print("### 2. min_confidence 分布（triggered + completed）")
    if cd["count"] > 0:
        print(f"  count  : {cd['count']}")
        print(f"  min    : {cd['min']}")
        print(f"  p25    : {cd['p25']}")
        print(f"  median : {cd['median']}")
        print(f"  p75    : {cd['p75']}")
        print(f"  max    : {cd['max']}")
        print(f"  (threshold = 0.90)")
    else:
        print("  （无数据）")
    print()

    # 3. decision matrix
    print("### 3. 决策矩阵（all_supported × confidence_ok × rescued）")
    print(f"  {'all_sup':>8s} {'conf_ok':>8s} {'rescued':>8s} {'cnt':>6s}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for r in stats["decision_matrix"]:
        print(
            f"  {str(r['all_supported']):>8s} "
            f"{str(r['confidence_ok']):>8s} "
            f"{str(r['rescued']):>8s} "
            f"{r['count']:>6d}"
        )
    print()

    # 4. trigger_status
    print("### 4. trigger_status 分布")
    for k, v in stats["trigger_status"].items():
        print(f"  {k:24s}: {v:5d}")
    print()

    # 5. execution_status
    es = stats["execution_status"]
    print("### 5. execution_status 分布")
    print(f"  completed  : {es['completed']}")
    print(f"  error      : {es['error']}")
    print(f"  error_rate : {es['error_rate']:.4f}")
    print()

    # 6. representative_claim_types
    print("### 6. representative_claim_type 分布")
    print("  ⚠️  这是 representative，不是 failure claim type")
    print(f"  {'type':24s} {'cnt':>6s} {'rescued':>8s} {'not_rescued':>12s}")
    print(f"  {'-'*24} {'-'*6} {'-'*8} {'-'*12}")
    for r in stats["representative_claim_types"]:
        print(
            f"  {r['type']:24s} "
            f"{r['count']:>6d} "
            f"{r['rescued']:>8d} "
            f"{r['not_rescued']:>12d}"
        )
    print()

    # 7. Bridge-level rescued=False 明细
    failed = stats["bridge_level_rescued_false"]
    print(f"### 7. Bridge-level rescued=False（{len(failed)} 个）")
    print("  来源: bridge_outcome_audit（无 JOIN）")
    print("  语义: 每次 try_rescue 调用独立记录")
    print()
    for b in failed:
        print(
            f"  --- bridge_id={b['bridge_id']} "
            f"scene={b['scene_id']} "
            f"(v{b['volume_num']} c{b['chapter_num']} s{b['scene_idx']}) ---"
        )
        print(
            f"      all_supported={b['all_supported']} "
            f"confidence_ok={b['confidence_ok']} "
            f"min_confidence={b['min_confidence']}"
        )
        if b["representative_claim_id"]:
            print(
                f"      representative: claim_id={b['representative_claim_id']} "
                f"type={b['representative_claim_type']}"
            )
        if b["representative_reason"]:
            print(f"      reason: {b['representative_reason'][:100]}")
        print()

    # 8. Claim-level 概览
    claim_all = stats.get("claim_level_all", [])
    print(f"### 8. Claim-level 概览（{len(claim_all)} 条）")
    print("  来源: validator_v2_audit（无 JOIN）")
    print()
    if claim_all:
        # 只打印统计摘要，不逐条
        from collections import Counter
        mode_cnt = Counter(c["mode"] for c in claim_all)
        final_cnt = Counter(c["final_verdict"] for c in claim_all)
        llm_cnt = Counter(
            (c["llm_raw_verdict"] or "not_called") for c in claim_all
        )
        print(f"  by mode:")
        for k, v in mode_cnt.most_common():
            print(f"    {k:14s}: {v}")
        print(f"  by final_verdict:")
        for k, v in final_cnt.most_common():
            print(f"    {k:14s}: {v}")
        print(f"  by llm_raw_verdict:")
        for k, v in llm_cnt.most_common():
            print(f"    {k:14s}: {v}")
    print()
    print("  完整明细见 JSON 字段: claim_level_all")
    print()


# ============================================================
# 主入口
# ============================================================

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="C3.4.3B.5 Bridge Outcome Replay"
    )
    parser.add_argument("--novel_id", default=None)
    args = parser.parse_args()

    try:
        pool = await _create_pool()
    except Exception as e:
        print(f"❌ 无法连接数据库: {type(e).__name__}: {e}")
        return 1

    try:
        where, wargs = await _build_where(args.novel_id)
        async with pool.acquire() as conn:
            stats = await _analyze(conn, where, wargs)
    finally:
        await pool.close()

    _print_report(stats)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"c3_4_3b_bridge_outcome_replay_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"JSON: {out_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))