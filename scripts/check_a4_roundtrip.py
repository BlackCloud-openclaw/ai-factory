#!/usr/bin/env python
"""
A-4 验收：验证 contract_data 可 round-trip 且符合预期结构。
"""

import asyncio
import asyncpg
import sys
import json
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.writing.planning_contract import PlanningContract

# 数据库连接参数（与你 docker-compose 一致）
DSN = "postgresql://woami:kali@localhost:5432/ai_factory"


async def check():
    conn = await asyncpg.connect(DSN)
    try:
        row = await conn.fetchrow("""
            SELECT 
                scene_id,
                contract_data,
                original_violations,
                original_passed
            FROM shadow_rewrite_log
            WHERE contract_data IS NOT NULL
            ORDER BY executed_at DESC
            LIMIT 1
        """)

        if not row:
            print("⚠️ 没有找到包含 contract_data 的记录")
            print("   请先跑 2-3 章生成新数据")
            return

        scene_id = row["scene_id"]
        contract_data_raw = row["contract_data"]
        violations_raw = row["original_violations"]
        passed = row["original_passed"]

        print(f"📊 检查场景: {scene_id}")
        print("=" * 60)

        # ========== 手动解析 JSONB（遵循项目既有模式） ==========
        # asyncpg 有时返回字符串而非已解析对象，统一使用 json.loads
        contract_dict = json.loads(contract_data_raw) if isinstance(contract_data_raw, str) else contract_data_raw
        violations = json.loads(violations_raw) if isinstance(violations_raw, str) else violations_raw
        # ==========================================================

        # 1. 恢复 PlanningContract
        try:
            contract = PlanningContract(**contract_dict)
            print(f"✅ contract 恢复成功")
            print(f"   scene_id: {contract.scene_id}")
            print(f"   intent.goal: {contract.intent.goal}")
            print(f"   state_changes 数量: {len(contract.observables.state_changes)}")
        except Exception as e:
            print(f"❌ contract 恢复失败: {e}")
            # 打印原始数据的前 200 字符帮助调试
            print(f"   原始 contract_data 前 200 字符: {str(contract_data_raw)[:200]}")
            return

        # 2. 检查 violations 结构
        print("\n📋 original_violations 结构:")
        print(f"   类型: {type(violations)}")
        if violations:
            print(f"   长度: {len(violations)}")
            print(f"   第一个元素类型: {type(violations[0])}")
            if isinstance(violations[0], dict):
                print(f"   第一个元素 keys: {list(violations[0].keys())}")
                print(f"   第一个元素 sample: {violations[0]}")
            elif isinstance(violations[0], str):
                print(f"   第一个元素 sample: {violations[0][:100]}")
        else:
            print("   (空)")

        # 3. 检查 passed
        print(f"\n🎯 original_passed: {passed}")

        print("\n✅ A-4 检查完成")
        print("   → contract_data 可 round-trip")
        print("   → 请将上面的 violations 结构反馈给开发者，用于编写 A-5")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(check())