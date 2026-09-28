#!/usr/bin/env python
"""
A-5 / E-2A: Offline Replay
验证 Shadow Validator 与 Production Validator 对同一 Original 的 PASS/FAIL 判定一致性。

验收标准：
- 10/10 MATCH → E-2A PASS → 允许进入 E-3
- 任何 MISMATCH → E-2A BLOCKED → 调查差异

注意：
- 只比较 passed，不比较 violations
- offline_missing 只作为诊断信息输出
- 不将 MISMATCH 归因于 writer_artifact.events，需调查实际输入差异
"""

import asyncio
import asyncpg
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.writing.validation.semantic_validator import SemanticValidator
from src.writing.planning_contract import PlanningContract

DSN = "postgresql://woami:kali@localhost:5432/ai_factory"


async def e2a_offline_replay():
    conn = await asyncpg.connect(DSN)
    try:
        rows = await conn.fetch("""
            SELECT 
                scene_id,
                original_text,
                contract_data,
                original_passed,
                executed_at
            FROM shadow_rewrite_log
            WHERE experiment_id = 'phase15.3.v1'
              AND status NOT IN ('skipped', 'llm_error', 'timeout')
              AND contract_data IS NOT NULL
              AND original_text IS NOT NULL
            ORDER BY executed_at DESC
            LIMIT 10
        """)

        if not rows:
            print("⚠️ 没有找到包含 contract_data 的新记录")
            print("   请先跑 2-3 章生成新数据")
            return

        print(f"📊 找到 {len(rows)} 条可重放的记录")
        print("=" * 70)

        validator = SemanticValidator()
        passed_count = 0
        failed_count = 0
        mismatch_scenes = []

        for row in rows:
            scene_id = row["scene_id"]
            runtime_passed = row["original_passed"]
            contract_data_raw = row["contract_data"]
            contract_dict = json.loads(contract_data_raw) if isinstance(contract_data_raw, str) else contract_data_raw
            text = row["original_text"]

            # 恢复 PlanningContract
            contract = PlanningContract(**contract_dict)

            # Offline 验证（SemanticValidator.validate 是同步方法，不需要 await）
            offline_result = validator.validate(contract, text)
            offline_passed = offline_result.passed
            offline_missing = offline_result.missing

            # 比较 passed
            if runtime_passed == offline_passed:
                status = "✅ MATCH"
                passed_count += 1
            else:
                status = "❌ MISMATCH"
                failed_count += 1
                mismatch_scenes.append(scene_id)

            print(f"{status} | {scene_id}")
            print(f"  Runtime passed: {runtime_passed}")
            print(f"  Offline passed: {offline_passed}")
            if offline_missing:
                print(f"  Offline missing: {offline_missing[:3]}... (total {len(offline_missing)})")
            print()

        print("=" * 70)
        print(f"📊 汇总: {passed_count} 匹配, {failed_count} 不匹配")

        if failed_count == 0:
            print("\n✅ E-2A 通过: 在当前 10 条样本上，Shadow Validator 与 Production Validator 的 PASS/FAIL 判定一致")
            print("   → 可以进入 E-3 Shadow Rewrite 实验")
        else:
            print(f"\n❌ E-2A 失败: {failed_count} 个场景不一致")
            print(f"   不匹配的场景: {', '.join(mismatch_scenes)}")
            print("\n   需要调查差异原因（contract 是否完全相同、original_text 是否完全相同、")
            print("   Production Validator 实际使用的输入是什么、passed 如何计算）")
            print("   暂不进入 E-3")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(e2a_offline_replay())