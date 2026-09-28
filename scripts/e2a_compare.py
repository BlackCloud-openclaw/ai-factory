#!/usr/bin/env python
"""
E-2A: Production Validator vs Shadow Validator 对照实验
只读分析，不修改任何数据或代码。
"""

import asyncio
import asyncpg
import os
import json
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径，以便导入 src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.writing.validation.semantic_validator import SemanticValidator
from src.writing.planning_contract import PlanningContract

# 数据库连接参数
DSN = "postgresql://woami:kali@localhost:5432/ai_factory"


async def e2a_compare():
    # 直接连接数据库
    try:
        conn = await asyncpg.connect(DSN)
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return

    try:
        # 1. 从数据库读取最近 10 个场景
        rows = await conn.fetch("""
            SELECT 
                scene_id,
                original_text,
                contract,
                original_passed,
                original_violations
            FROM shadow_rewrite_log
            WHERE experiment_id = 'phase15.3.v1'
              AND status NOT IN ('skipped', 'llm_error', 'timeout')
              AND original_text IS NOT NULL
              AND original_text != ''
            ORDER BY executed_at DESC
            LIMIT 10
        """)

        if not rows:
            print("⚠️ 没有找到有效样本")
            return

        print(f"📊 找到 {len(rows)} 个场景用于 E-2A 对照")
        print("=" * 70)

        # 2. 初始化 Shadow Validator
        validator = SemanticValidator()

        # 3. 逐场景对比
        results = {
            "both_pass": 0,
            "both_fail": 0,
            "prod_pass_shadow_fail": 0,
            "prod_fail_shadow_pass": 0,
        }

        detailed = []

        for row in rows:
            scene_id = row["scene_id"]
            production_passed = row["original_passed"]
            production_violations = row["original_violations"]
            contract = PlanningContract(**row["contract"])
            text = row["original_text"]

            # Shadow Validator 验证
            shadow_result = validator.validate(contract, text)
            shadow_passed = shadow_result.passed
            shadow_violations = [e for e in shadow_result.missing] if shadow_result.missing else []

            # 分类
            if production_passed and shadow_passed:
                category = "🟢 BOTH_PASS"
                results["both_pass"] += 1
            elif not production_passed and not shadow_passed:
                category = "⚪ BOTH_FAIL"
                results["both_fail"] += 1
            elif production_passed and not shadow_passed:
                category = "🔴 PROD_PASS_SHADOW_FAIL"
                results["prod_pass_shadow_fail"] += 1
            else:
                category = "🟡 PROD_FAIL_SHADOW_PASS"
                results["prod_fail_shadow_pass"] += 1

            detailed.append({
                "scene_id": scene_id,
                "category": category,
                "production_passed": production_passed,
                "shadow_passed": shadow_passed,
                "production_violations": production_violations,
                "shadow_violations": shadow_violations,
            })

            print(f"{category} | {scene_id}")
            print(f"  Production: passed={production_passed}, violations={len(production_violations) if production_violations else 0}")
            print(f"  Shadow:    passed={shadow_passed}, violations={len(shadow_violations)}")
            print()

        # 4. 汇总
        print("=" * 70)
        print("📊 E-2A 汇总:")
        print(f"  🟢 Both PASS: {results['both_pass']}")
        print(f"  ⚪ Both FAIL: {results['both_fail']}")
        print(f"  🔴 PROD PASS / Shadow FAIL: {results['prod_pass_shadow_fail']}  ← 关键指标")
        print(f"  🟡 PROD FAIL / Shadow PASS: {results['prod_fail_shadow_pass']}")

        total = sum(results.values())
        if total > 0:
            print(f"\n  一致率: {(results['both_pass'] + results['both_fail']) / total * 100:.1f}%")

        # 5. 如果存在 PROD_PASS_SHADOW_FAIL，显示详情
        if results["prod_pass_shadow_fail"] > 0:
            print("\n🔴 关键发现: Production PASS 但 Shadow FAIL 的场景:")
            for d in detailed:
                if d["category"] == "🔴 PROD_PASS_SHADOW_FAIL":
                    print(f"  - {d['scene_id']}")
                    print(f"    Production violations: {d['production_violations']}")
                    print(f"    Shadow violations: {d['shadow_violations']}")
        else:
            print("\n✅ 没有发现 Production PASS → Shadow FAIL 的降级情况")

        # 6. 给出建议
        if results["prod_pass_shadow_fail"] == 0:
            print("\n✅ E-2A 初步结论: Shadow Validator 在采样样本中没有漏判 Production Validator 通过的内容。")
            print("   可以进入下一步讨论 E-2B 或直接进入 E-3。")
        else:
            print("\n⚠️ E-2A 发现 Shadow Validator 存在假阴性。")
            print("   需要分析具体差异，确定是否可以接受，或需要调整 Validator 配置。")
            print("   不建议直接进入 E-3，先分析差异原因。")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(e2a_compare())