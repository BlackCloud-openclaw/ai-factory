#!/usr/bin/env python
"""
C3.4.2c 独立验证：audit_writer 能正确写入 validator_v2_audit 表。

不依赖 Validator / LLM / 全局 db pool，自己建 asyncpg pool。
验证三态语义（None / [] / [N]）与 LLM 判定降级链被正确持久化。

数据库连接配置（与 docker-compose 一致，可通过环境变量覆盖）：
    POSTGRES_HOST     默认 localhost
    POSTGRES_PORT     默认 5432
    POSTGRES_DB       默认 ai_factory
    POSTGRES_USER     默认 woami
    POSTGRES_PASSWORD 默认 kali
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import asyncpg

from src.writing.validation_v2.models import (
    ValidationResultV2,
    Verdict,
    MatchLayer,
    SemanticJudgement,
)
from src.writing.validation_v2.audit_writer import (
    AuditContext,
    record_audit_batch,
)


def _make_results():
    """构造 4 条覆盖三态语义 + LLM 降级链的样本"""

    # 1. EXACT 短路：ret_ids=None
    r1 = ValidationResultV2(
        claim_id="test_exact_001",
        state_change_type="realm_change",
        verdict=Verdict.SUPPORTED,
        matched_layer=MatchLayer.EXACT,
        confidence=1.0,
        reason="Realm structural exact match",
        evidence_candidates_found=True,
        structural_check="exact_match",
        fallback_applied=False,
        retrieved_evidence_count=0,
        retrieved_evidence_ids=None,
    )

    # 2. MISMATCH：ret_ids=[]
    r2 = ValidationResultV2(
        claim_id="test_mismatch_002",
        state_change_type="realm_change",
        verdict=Verdict.INSUFFICIENT,
        matched_layer=MatchLayer.NONE,
        confidence=0.0,
        reason="Realm structural mismatch",
        evidence_candidates_found=False,
        structural_check="mismatch",
        fallback_applied=True,
        retrieved_evidence_count=0,
        retrieved_evidence_ids=[],
    )

    # 3. LLM SUPPORTED 被降级：ret_ids=[N]
    r3 = ValidationResultV2(
        claim_id="test_downgrade_003",
        state_change_type="realm_change",
        verdict=Verdict.INSUFFICIENT,
        matched_layer=MatchLayer.NONE,
        confidence=0.0,
        reason="Semantic supports but no structural match",
        evidence_candidates_found=True,
        structural_check="no_event",
        fallback_applied=True,
        judgement=SemanticJudgement(
            verdict=Verdict.SUPPORTED,
            confidence=0.9,
            reason="证据表明境界已完成",
            evidence_ids=["txt_0"],
        ),
        retrieved_evidence_count=1,
        retrieved_evidence_ids=["txt_0"],
    )

    # 4. LLM CONTRADICTED 被采纳：ret_ids=[N]
    r4 = ValidationResultV2(
        claim_id="test_contradicted_004",
        state_change_type="plot_flag",
        verdict=Verdict.CONTRADICTED,
        matched_layer=MatchLayer.SEMANTIC,
        confidence=0.95,
        reason="证据明确否定",
        evidence_candidates_found=True,
        structural_check=None,
        fallback_applied=False,
        judgement=SemanticJudgement(
            verdict=Verdict.CONTRADICTED,
            confidence=0.95,
            reason="证据明确否定",
            evidence_ids=["txt_0"],
        ),
        retrieved_evidence_count=1,
        retrieved_evidence_ids=["txt_0"],
    )

    return [r1, r2, r3, r4]


async def _create_pool() -> asyncpg.Pool:
    """创建独立的 asyncpg pool（不依赖全局 src.db 状态）"""
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


async def main() -> int:
    # ---- 建 pool ----
    try:
        pool = await _create_pool()
    except Exception as e:
        print(f"❌ 无法连接数据库: {type(e).__name__}: {e}")
        return 1

    try:
        results = _make_results()
        context = AuditContext(
            novel_id="__audit_writer_test__",
            volume_num=0,
            chapter_num=0,
            scene_idx=0,
            scene_id="test_scene",
            source="c3_3b_eval",
            mode="shadow_only",
        )

        async with pool.acquire() as conn:
            # 清理测试数据
            await conn.execute(
                "DELETE FROM validator_v2_audit WHERE novel_id = $1",
                context.novel_id,
            )

            # 写入
            n = await record_audit_batch(conn, results, context)
            print(f"✅ 写入 {n} 条记录（期望 {len(results)}）")

            # 回读验证
            rows = await conn.fetch(
                """
                SELECT claim_id, structural_check,
                       retrieved_evidence_count, retrieved_evidence_ids,
                       llm_invoked, llm_raw_verdict, final_verdict,
                       fallback_applied
                FROM validator_v2_audit
                WHERE novel_id = $1
                ORDER BY claim_id
                """,
                context.novel_id,
            )

        print()
        hdr = (
            f"{'claim_id':22s} {'struct':11s} {'ret_c':5s} "
            f"{'ret_ids':10s} {'llm':5s} {'llm_v':14s} "
            f"{'final':14s} {'fb':5s}"
        )
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            ret_ids = r["retrieved_evidence_ids"]
            # asyncpg 默认把 JSONB 解为 str（若 codec 未配置）；为安全兼容两种形态
            if ret_ids is None:
                ret_ids_str = "None"
            elif isinstance(ret_ids, str):
                import json as _json
                ret_ids_str = f"[{len(_json.loads(ret_ids))}]"
            else:
                ret_ids_str = f"[{len(ret_ids)}]"

            llm_v = r["llm_raw_verdict"] or "-"
            print(
                f"{r['claim_id']:22s} "
                f"{r['structural_check'] or '-':11s} "
                f"{str(r['retrieved_evidence_count']):5s} "
                f"{ret_ids_str:10s} "
                f"{str(r['llm_invoked']):5s} "
                f"{llm_v:14s} "
                f"{r['final_verdict']:14s} "
                f"{str(r['fallback_applied']):5s}"
            )

        # ---- 断言 ----
        assert len(rows) == 4, f"Expected 4 rows, got {len(rows)}"

        by_claim = {r["claim_id"]: r for r in rows}

        # 三态语义
        assert by_claim["test_exact_001"]["retrieved_evidence_ids"] is None, \
            "EXACT 短路应保留 None"
        # mismatch 行 retrieved_evidence_ids 应为 "[]" 或 []
        mismatch_val = by_claim["test_mismatch_002"]["retrieved_evidence_ids"]
        if isinstance(mismatch_val, str):
            import json as _json
            assert _json.loads(mismatch_val) == [], "MISMATCH 应为 []"
        else:
            assert mismatch_val == [], "MISMATCH 应为 []"

        # LLM 降级链
        assert by_claim["test_downgrade_003"]["llm_raw_verdict"] == "SUPPORTED"
        assert by_claim["test_downgrade_003"]["final_verdict"] == "INSUFFICIENT"
        assert by_claim["test_downgrade_003"]["fallback_applied"] is True

        # LLM 采纳
        assert by_claim["test_contradicted_004"]["llm_raw_verdict"] == "CONTRADICTED"
        assert by_claim["test_contradicted_004"]["final_verdict"] == "CONTRADICTED"
        assert by_claim["test_contradicted_004"]["fallback_applied"] is False

        print()
        print("✅ 三态语义 + LLM 降级链全部正确")

        # 清理
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM validator_v2_audit WHERE novel_id = $1",
                context.novel_id,
            )
        print("✅ 测试数据已清理")
        return 0

    finally:
        await pool.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))