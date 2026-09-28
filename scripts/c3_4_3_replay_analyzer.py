#!/usr/bin/env python
"""
C3.4.3A: Validator V2 Replay Baseline Analyzer

只读 validator_v2_audit 表，输出 baseline 报告。
不修改任何生产代码。

设计原则:
- 完全独立（不依赖 src.db 全局 pool）
- 只读（SELECT only）
- 只覆盖 Validator V2 claim-level 层
  （Rescue Outcome 层留待 C3.4.3B）
- baseline analyzer 不做人为采样限制；
  之后若需要抽样，另设 --latest N 或时间窗口参数

Retriever 三态（严格区分）:
- not_called: retrieved_evidence_ids IS NULL
              （EXACT / ALIAS / realm EXACT_MATCH 短路路径）
- empty:      retrieved_evidence_ids = '[]'::jsonb
              （Retriever 已调用但未命中）
- hit:        retrieved_evidence_ids 为非空数组
              （Retriever 命中 N 条）

输出:
  - stdout: 可读摘要
  - reports/c3_4_3a_replay_baseline_{ts}.json: machine-readable baseline

用法:
  python scripts/c3_4_3_replay_analyzer.py
  python scripts/c3_4_3_replay_analyzer.py --novel_id simple_long_novel_001
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import asyncpg


SCHEMA_VERSION = "c3_4_3a.v1"


# ============================================================
# 数据库连接
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
# 分析查询
# ============================================================

async def _build_where(novel_id: str | None) -> tuple[str, list]:
    if novel_id:
        return "WHERE novel_id = $1", [novel_id]
    return "", []


async def _analyze(conn, where: str, args: list) -> dict:
    stats: dict = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(),
        "filters": {
            "novel_id": args[0] if args else None,
        },
    }

    # ------------------------------------------------------------
    # 0. 总量
    # ------------------------------------------------------------
    row = await conn.fetchrow(
        f"SELECT COUNT(*) AS c FROM validator_v2_audit {where}", *args
    )
    stats["total_rows"] = row["c"]

    if stats["total_rows"] == 0:
        return stats

    # ------------------------------------------------------------
    # 1. mode 分布
    # ------------------------------------------------------------
    rows = await conn.fetch(f"""
        SELECT mode, COUNT(*) AS c
        FROM validator_v2_audit {where}
        GROUP BY mode
        ORDER BY c DESC
    """, *args)
    stats["by_mode"] = {r["mode"]: r["c"] for r in rows}

    # ------------------------------------------------------------
    # 2. state_change_type 分布
    # ------------------------------------------------------------
    rows = await conn.fetch(f"""
        SELECT state_change_type, COUNT(*) AS c
        FROM validator_v2_audit {where}
        GROUP BY state_change_type
        ORDER BY c DESC
    """, *args)
    stats["by_state_change_type"] = {r["state_change_type"]: r["c"] for r in rows}

    # ------------------------------------------------------------
    # 3. retrieval 三态
    #    not_called: NULL
    #    empty     : '[]'
    #    hit       : 非空数组
    # ------------------------------------------------------------
    rows = await conn.fetch(f"""
        SELECT
            CASE
                WHEN retrieved_evidence_ids IS NULL THEN 'not_called'
                WHEN retrieved_evidence_ids::jsonb = '[]'::jsonb THEN 'empty'
                ELSE 'hit'
            END AS retrieval_state,
            COUNT(*) AS c
        FROM validator_v2_audit {where}
        GROUP BY retrieval_state
        ORDER BY c DESC
    """, *args)
    stats["by_retrieval_state"] = {r["retrieval_state"]: r["c"] for r in rows}

    # ------------------------------------------------------------
    # 4. LLM 状态
    #    not_called: llm_invoked=FALSE
    #    parse_error: llm_raw_reason IN ('Parse error', 'Invalid JSON')
    #    else: llm_raw_verdict
    # ------------------------------------------------------------
    rows = await conn.fetch(f"""
        SELECT
            CASE
                WHEN llm_invoked = FALSE THEN 'not_called'
                WHEN llm_raw_reason IN ('Parse error', 'Invalid JSON') THEN 'parse_error'
                ELSE llm_raw_verdict
            END AS llm_state,
            COUNT(*) AS c
        FROM validator_v2_audit {where}
        GROUP BY llm_state
        ORDER BY c DESC
    """, *args)
    stats["by_llm_state"] = {r["llm_state"]: r["c"] for r in rows}

    # ------------------------------------------------------------
    # 5. final_verdict 分布
    # ------------------------------------------------------------
    rows = await conn.fetch(f"""
        SELECT final_verdict, COUNT(*) AS c
        FROM validator_v2_audit {where}
        GROUP BY final_verdict
        ORDER BY c DESC
    """, *args)
    stats["by_final_verdict"] = {r["final_verdict"]: r["c"] for r in rows}

    # ------------------------------------------------------------
    # 6. divergence (llm_raw != final，仅 llm_invoked=TRUE)
    # ------------------------------------------------------------
    row = await conn.fetchrow(f"""
        SELECT COUNT(*) AS c
        FROM validator_v2_audit {where}
        {'AND' if where else 'WHERE'} llm_invoked = TRUE
          AND llm_raw_verdict IS DISTINCT FROM final_verdict
    """, *args)
    stats["divergence_count"] = row["c"]

    # 6b. 具体 divergence 分布
    rows = await conn.fetch(f"""
        SELECT llm_raw_verdict, final_verdict, COUNT(*) AS c
        FROM validator_v2_audit {where}
        {'AND' if where else 'WHERE'} llm_invoked = TRUE
          AND llm_raw_verdict IS DISTINCT FROM final_verdict
        GROUP BY llm_raw_verdict, final_verdict
        ORDER BY c DESC
    """, *args)
    stats["divergence_distribution"] = [
        {"llm_raw_verdict": r["llm_raw_verdict"],
         "final_verdict": r["final_verdict"],
         "count": r["c"]}
        for r in rows
    ]

    # ------------------------------------------------------------
    # 7. 重点观察交叉
    # ------------------------------------------------------------

    # 7.1 LLM SUPPORTED → final INSUFFICIENT（B2-2 核心不变量）
    row = await conn.fetchrow(f"""
        SELECT COUNT(*) AS c
        FROM validator_v2_audit {where}
        {'AND' if where else 'WHERE'} llm_invoked = TRUE
          AND llm_raw_verdict = 'SUPPORTED'
          AND final_verdict = 'INSUFFICIENT'
    """, *args)
    stats["obs_7_1_llm_supported_final_insufficient"] = row["c"]

    # 7.2 LLM CONTRADICTED → final CONTRADICTED（正常采纳）
    row = await conn.fetchrow(f"""
        SELECT COUNT(*) AS c
        FROM validator_v2_audit {where}
        {'AND' if where else 'WHERE'} llm_invoked = TRUE
          AND llm_raw_verdict = 'CONTRADICTED'
          AND final_verdict = 'CONTRADICTED'
    """, *args)
    stats["obs_7_2_llm_contradicted_final_contradicted"] = row["c"]

    # 7.3 Retriever 命中 → LLM INSUFFICIENT
    row = await conn.fetchrow(f"""
        SELECT COUNT(*) AS c
        FROM validator_v2_audit {where}
        {'AND' if where else 'WHERE'}
            retrieved_evidence_ids IS NOT NULL
            AND retrieved_evidence_ids::jsonb != '[]'::jsonb
            AND llm_invoked = TRUE
            AND llm_raw_verdict = 'INSUFFICIENT'
    """, *args)
    stats["obs_7_3_retriever_hit_llm_insufficient"] = row["c"]

    # 7.4 Retriever 空 → final INSUFFICIENT
    row = await conn.fetchrow(f"""
        SELECT COUNT(*) AS c
        FROM validator_v2_audit {where}
        {'AND' if where else 'WHERE'}
            retrieved_evidence_ids IS NOT NULL
            AND retrieved_evidence_ids::jsonb = '[]'::jsonb
            AND final_verdict = 'INSUFFICIENT'
    """, *args)
    stats["obs_7_4_retriever_empty_final_insufficient"] = row["c"]

    # 7.5 EXACT/ALIAS 短路（llm_invoked=FALSE AND final=SUPPORTED）
    row = await conn.fetchrow(f"""
        SELECT COUNT(*) AS c
        FROM validator_v2_audit {where}
        {'AND' if where else 'WHERE'} llm_invoked = FALSE
          AND final_verdict = 'SUPPORTED'
    """, *args)
    stats["obs_7_5_short_circuit_supported"] = row["c"]

    # ------------------------------------------------------------
    # 8. mode × type × verdict 全景（供深度 diff）
    # ------------------------------------------------------------
    rows = await conn.fetch(f"""
        SELECT mode, state_change_type, final_verdict, COUNT(*) AS c
        FROM validator_v2_audit {where}
        GROUP BY mode, state_change_type, final_verdict
        ORDER BY mode, state_change_type, c DESC
    """, *args)
    stats["cross_mode_type_final"] = [
        {"mode": r["mode"],
         "state_change_type": r["state_change_type"],
         "final_verdict": r["final_verdict"],
         "count": r["c"]}
        for r in rows
    ]

    # ------------------------------------------------------------
    # 9. 一致性检查
    #    llm_invoked=FALSE 但 Retriever 明明命中了 → 不应发生
    # ------------------------------------------------------------
    row = await conn.fetchrow(f"""
        SELECT COUNT(*) AS c
        FROM validator_v2_audit {where}
        {'AND' if where else 'WHERE'} llm_invoked = FALSE
          AND retrieved_evidence_ids IS NOT NULL
          AND retrieved_evidence_ids::jsonb != '[]'::jsonb
    """, *args)
    stats["consistency_llm_not_invoked_but_evidence_hit"] = row["c"]

    return stats


# ============================================================
# 输出
# ============================================================

def _pct(part: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{part / total * 100:.1f}%"


def _print_report(stats: dict) -> None:
    total = stats["total_rows"]

    print()
    print("=" * 64)
    print("C3.4.3A: Validator V2 Replay Baseline")
    print("=" * 64)
    print(f"Schema     : {stats['schema_version']}")
    print(f"Generated  : {stats['generated_at']}")
    print(f"Filters    : {stats['filters']}")
    print(f"Total rows : {total}")
    print()

    if total == 0:
        print("⚠️  无数据。")
        return

    # 1. mode
    print("### 1. Mode 分布")
    for k, v in stats["by_mode"].items():
        print(f"  {k:14s}: {v:5d}  ({_pct(v, total)})")
    print()

    # 2. state_change_type
    print("### 2. state_change_type 分布")
    for k, v in stats["by_state_change_type"].items():
        print(f"  {k:22s}: {v:5d}  ({_pct(v, total)})")
    print()

    # 3. retrieval
    print("### 3. Retrieval 三态（严格区分）")
    label = {
        "not_called": "Retriever 未调用 (NULL)",
        "empty": "Retriever 空命中 ([])",
        "hit": "Retriever 命中 ([N])",
    }
    for k in ("not_called", "empty", "hit"):
        if k in stats["by_retrieval_state"]:
            v = stats["by_retrieval_state"][k]
            print(f"  {label.get(k, k):30s}: {v:5d}  ({_pct(v, total)})")
    print()

    # 4. llm
    print("### 4. LLM 状态")
    for k, v in stats["by_llm_state"].items():
        print(f"  {k:22s}: {v:5d}  ({_pct(v, total)})")
    print()

    # 5. final
    print("### 5. Final verdict 分布")
    for k, v in stats["by_final_verdict"].items():
        print(f"  {k:22s}: {v:5d}  ({_pct(v, total)})")
    print()

    # 6. divergence
    print("### 6. Divergence (llm_raw != final, llm_invoked=TRUE)")
    print(f"  总数: {stats['divergence_count']}")
    if stats["divergence_distribution"]:
        for d in stats["divergence_distribution"]:
            print(
                f"    {d['llm_raw_verdict']:14s} → "
                f"{d['final_verdict']:14s} : {d['count']}"
            )
    print()

    # 7. 重点观察
    print("### 7. 重点观察")
    print(f"  7.1 LLM SUPPORTED → final INSUFFICIENT    : "
          f"{stats['obs_7_1_llm_supported_final_insufficient']}")
    print(f"  7.2 LLM CONTRADICTED → final CONTRADICTED: "
          f"{stats['obs_7_2_llm_contradicted_final_contradicted']}")
    print(f"  7.3 Retriever 命中 → LLM INSUFFICIENT     : "
          f"{stats['obs_7_3_retriever_hit_llm_insufficient']}")
    print(f"  7.4 Retriever 空 → final INSUFFICIENT     : "
          f"{stats['obs_7_4_retriever_empty_final_insufficient']}")
    print(f"  7.5 EXACT/ALIAS 短路 → SUPPORTED          : "
          f"{stats['obs_7_5_short_circuit_supported']}")
    print()

    # 8. 一致性检查
    print("### 8. 一致性检查")
    n = stats["consistency_llm_not_invoked_but_evidence_hit"]
    icon = "✅" if n == 0 else "⚠️"
    print(f"  {icon} llm_invoked=FALSE 但有命中证据: {n}")
    print()

    # 9. mode × type × final 全景（截断显示）
    print("### 9. mode × type × final 全景")
    print(f"  {'mode':14s} {'state_change_type':22s} {'final':14s} {'count':>6s}")
    print(f"  {'-'*14} {'-'*22} {'-'*14} {'-'*6}")
    for r in stats["cross_mode_type_final"]:
        print(
            f"  {r['mode']:14s} "
            f"{r['state_change_type']:22s} "
            f"{r['final_verdict']:14s} "
            f"{r['count']:>6d}"
        )
    print()


# ============================================================
# 主入口
# ============================================================

async def main() -> int:
    parser = argparse.ArgumentParser(description="C3.4.3A Replay Baseline Analyzer")
    parser.add_argument("--novel_id", default=None, help="只分析指定 novel_id")
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

    # ---- 保存 JSON ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"c3_4_3a_replay_baseline_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Baseline JSON: {out_path}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))