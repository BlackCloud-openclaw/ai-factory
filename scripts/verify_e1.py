#!/usr/bin/env python
"""
E-1 轻量验收脚本
不依赖 src 模块，只连接数据库查询统计。
人工核对日志中的 validation_result["passed"] 是否一致。
"""
import asyncio
import asyncpg
import os
from datetime import datetime


async def main():
    # 数据库连接配置（与你的 .env 或 docker-compose 一致）
    dsn = os.getenv(
        "DATABASE_URL",
        "postgresql://woami:kali@localhost:5432/ai_factory"
    )

    conn = await asyncpg.connect(dsn)
    try:
        # 1. 检查 E-1 补丁后的数据
        rows = await conn.fetch("""
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE original_passed = true) as pass_count,
                COUNT(*) FILTER (WHERE original_passed = false) as fail_count,
                COUNT(*) FILTER (WHERE original_passed IS NULL) as null_count,
                MIN(executed_at) as first_sample,
                MAX(executed_at) as last_sample
            FROM shadow_rewrite_log
            WHERE experiment_id = 'phase15.3.v1'
              AND executed_at >= '2026-08-16 10:00:00'  -- ⚠️ 调整为你的补丁应用时间
              AND status NOT IN ('skipped', 'llm_error', 'timeout');
        """)

        row = rows[0]
        total = row["total"]
        pass_count = row["pass_count"]
        fail_count = row["fail_count"]
        null_count = row["null_count"]

        print(f"📊 E-1 样本统计 (时间过滤后):")
        print(f"  总样本: {total}")
        print(f"  PASS: {pass_count}")
        print(f"  FAIL: {fail_count}")
        print(f"  NULL: {null_count}")

        if total == 0:
            print("⚠️  没有样本，请确认时间过滤条件是否正确。")
            return

        if null_count > 0:
            print("⚠️  发现 NULL 值，检查数据完整性。")

        # 2. 列出最近 5 个样本，供人工核对
        samples = await conn.fetch("""
            SELECT scene_id, original_passed, executed_at
            FROM shadow_rewrite_log
            WHERE experiment_id = 'phase15.3.v1'
              AND executed_at >= '2026-08-16 10:00:00'
              AND status NOT IN ('skipped', 'llm_error', 'timeout')
            ORDER BY executed_at DESC
            LIMIT 5;
        """)

        print("\n🔍 最近 5 个样本 (请与 logs/ai_factory.log 中的 validation_result 核对):")
        for s in samples:
            print(f"  {s['scene_id']}: original_passed={s['original_passed']}  at {s['executed_at']}")

        # 3. 手动核对提示
        if pass_count > 0 and fail_count > 0:
            print("\n✅ E-1 初步通过：样本中同时存在 PASS 和 FAIL。")
            print("📌 请随机选取 1 个 PASS 和 1 个 FAIL 的 scene_id，")
            print("   在 logs/ai_factory.log 中搜索 'validation_result' 确认 passed 值是否一致。")
            print("   若一致，则 E-1 正式通过。")
        else:
            print("\n⚠️  E-1 尚未满足验收条件：需要同时包含 PASS 和 FAIL 样本。")
            print("   如果所有样本都是 FAIL，请检查生产 Validator 是否对当前章节都判定失败，或确认补丁是否生效。")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())