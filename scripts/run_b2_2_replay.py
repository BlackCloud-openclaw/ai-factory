#!/usr/bin/env python
"""
B2-2A.2: 改进版 Replay

- 修复 ClaimBuilder unknown 诊断
- 改进 EvidenceRetriever（窗口 + 分块）
- 增加 claim_type_distribution
- 增加 evidence_candidates_found / evidence_status 统计
- 保持 Shadow-only
"""

import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict

from src.db import init_db_pool, get_db_pool
from src.writing.validation_v2.validator import ValidationV2
from src.writing.validation_v2.models import MatchLayer, Verdict


async def fetch_samples(limit: int = 100) -> List[Dict[str, Any]]:
    await init_db_pool()
    pool = get_db_pool()
    if not pool:
        print("❌ 无法获取数据库连接")
        return []

    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                id,
                scene_id,
                original_text,
                rewritten_text,
                original_passed,
                rewritten_passed,
                contract_data,
                writer_events,
                original_violations,
                rewritten_violations
            FROM shadow_rewrite_log
            WHERE original_passed = false
              AND rewritten_passed = false
              AND status = 'validation_failed'
              AND original_text IS NOT NULL
            ORDER BY executed_at DESC
            LIMIT $1
        """, limit)

        samples = []
        for r in rows:
            contract_data = r["contract_data"]
            if isinstance(contract_data, str):
                try:
                    contract_data = json.loads(contract_data)
                except:
                    contract_data = {}

            writer_events = r["writer_events"]
            if isinstance(writer_events, str):
                try:
                    writer_events = json.loads(writer_events)
                except:
                    writer_events = []

            samples.append({
                "id": r["id"],
                "scene_id": r["scene_id"],
                "original_text": r["original_text"],
                "rewritten_text": r["rewritten_text"],
                "original_passed": r["original_passed"],
                "rewritten_passed": r["rewritten_passed"],
                "contract_data": contract_data,
                "writer_events": writer_events or [],
                "original_violations": r["original_violations"],
                "rewritten_violations": r["rewritten_violations"],
            })
        return samples


async def main():
    limit = 100
    samples = await fetch_samples(limit)
    print(f"📊 获取 {len(samples)} 个样本")

    if not samples:
        print("❌ 没有样本")
        return

    validator = ValidationV2()
    all_results = []

    # 统计变量
    total_claims = 0
    claim_type_distribution = defaultdict(int)
    verdict_distribution = defaultdict(int)
    evidence_found_count = 0
    evidence_not_found_count = 0
    evidence_status_distribution = defaultdict(int)

    for idx, sample in enumerate(samples):
        sample_id = sample["id"]
        scene_id = sample["scene_id"]
        contract = sample.get("contract_data", {})
        writer_events = sample.get("writer_events", [])
        original_text = sample.get("original_text", "")

        if not contract:
            continue

        results = await validator.validate_contract(contract, writer_events, original_text)

        if results:
            sample_summary = {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "state_changes": [r.to_dict() for r in results],
                "summary": {
                    "total": len(results),
                    "exact": sum(1 for r in results if r.matched_layer == MatchLayer.EXACT),
                    "alias": sum(1 for r in results if r.matched_layer == MatchLayer.ALIAS),
                    "semantic_supported": sum(1 for r in results if r.matched_layer == MatchLayer.SEMANTIC and r.verdict == Verdict.SUPPORTED),
                    "semantic_contradicted": sum(1 for r in results if r.matched_layer == MatchLayer.SEMANTIC and r.verdict == Verdict.CONTRADICTED),
                    "insufficient": sum(1 for r in results if r.matched_layer == MatchLayer.NONE),
                }
            }
            all_results.append(sample_summary)

            # 累加统计
            for r in results:
                total_claims += 1
                claim_type_distribution[r.state_change_type] += 1
                verdict_distribution[r.verdict.value] += 1
                if r.evidence_candidates_found:
                    evidence_found_count += 1
                else:
                    evidence_not_found_count += 1
                evidence_status_distribution[r.evidence_status] += 1

    # 生成报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "sample_count": len(all_results),
        "total_claims": total_claims,
        "claim_type_distribution": dict(claim_type_distribution),
        "verdict_distribution": dict(verdict_distribution),
        "evidence_candidates": {
            "found": evidence_found_count,
            "not_found": evidence_not_found_count,
            "found_percentage": round(evidence_found_count / total_claims * 100, 1) if total_claims else 0,
        },
        "evidence_status_distribution": dict(evidence_status_distribution),
        "percentages": {
            "exact": round(claim_type_distribution.get("exact", 0) / total_claims * 100, 1) if total_claims else 0,
            "alias": round(claim_type_distribution.get("alias", 0) / total_claims * 100, 1) if total_claims else 0,
            "semantic_supported": round(verdict_distribution.get("SUPPORTED", 0) / total_claims * 100, 1) if total_claims else 0,
            "semantic_contradicted": round(verdict_distribution.get("CONTRADICTED", 0) / total_claims * 100, 1) if total_claims else 0,
            "insufficient": round(verdict_distribution.get("INSUFFICIENT", 0) / total_claims * 100, 1) if total_claims else 0,
        },
        "coverage": round((verdict_distribution.get("SUPPORTED", 0) + verdict_distribution.get("CONTRADICTED", 0)) / total_claims * 100, 1) if total_claims else 0,
        "details": all_results[:20],
    }

    output_path = project_root / f"reports/b2_2_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n📊 统计报告已保存到: {output_path}")
    print(f"\n📈 汇总:")
    print(f"  总 Claim 数: {total_claims}")
    print(f"  类型分布:")
    for t, c in sorted(claim_type_distribution.items(), key=lambda x: -x[1]):
        print(f"    {t}: {c} ({c/total_claims*100:.1f}%)")
    print(f"\n  判决分布:")
    for v, c in sorted(verdict_distribution.items(), key=lambda x: -x[1]):
        print(f"    {v}: {c} ({c/total_claims*100:.1f}%)")
    print(f"\n  证据检索:")
    print(f"    找到候选: {evidence_found_count} ({report['evidence_candidates']['found_percentage']:.1f}%)")
    print(f"    未找到候选: {evidence_not_found_count}")
    print(f"\n  证据状态分布:")
    for s, c in sorted(evidence_status_distribution.items(), key=lambda x: -x[1]):
        print(f"    {s}: {c} ({c/total_claims*100:.1f}%)")
    print(f"\n  覆盖率: {report['coverage']:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())