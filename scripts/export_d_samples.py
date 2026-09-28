#!/usr/bin/env python
"""
导出最近30个D象限FAIL样本用于Phase 15.7-B2-V0审计
"""

import sys
import os
import asyncio
import json
from pathlib import Path

# ---- 第1步：确定项目根目录并添加至 sys.path ----
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ---- 第2步：加载 .env 文件（必须在导入 src 之前） ----
from dotenv import load_dotenv
dotenv_path = project_root / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)
    print(f"✅ 已加载环境变量: {dotenv_path}")
else:
    print(f"⚠️ 未找到 .env 文件: {dotenv_path}，将尝试使用系统环境变量")

# ---- 第3步：现在可以安全导入 src 模块 ----
from src.db import get_db_pool, init_db_pool
from src.config import config


async def main():
    print("🔍 正在初始化数据库连接池...")
    await init_db_pool()
    
    pool = get_db_pool()
    if not pool:
        print("❌ 无法获取数据库连接池，请检查数据库配置和网络")
        return

    print("✅ 数据库连接池获取成功，正在查询样本...")

    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT 
                id,
                scene_id,
                original_text,
                rewritten_text,
                original_passed,
                rewritten_passed,
                status,
                contract_data,
                writer_events,
                original_violations,
                rewritten_violations,
                executed_at
            FROM shadow_rewrite_log
            WHERE original_passed = false 
              AND rewritten_passed = false
              AND status = 'success'
              AND original_text IS NOT NULL
            ORDER BY executed_at DESC
            LIMIT 30
        """)

        if not rows:
            print("⚠️ 没有找到符合条件的样本（D象限，status=success）")
            return

        samples = []
        for r in rows:
            samples.append({
                "id": r["id"],
                "scene_id": r["scene_id"],
                "original_text": r["original_text"],
                "rewritten_text": r["rewritten_text"],
                "original_passed": r["original_passed"],
                "rewritten_passed": r["rewritten_passed"],
                "contract_data": r["contract_data"],
                "writer_events": r["writer_events"],
                "original_violations": r["original_violations"],
                "rewritten_violations": r["rewritten_violations"],
                "executed_at": str(r["executed_at"]) if r["executed_at"] else None,
            })

        # 输出到项目根目录
        output_path = project_root / "d_quadrant_30_samples.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        print(f"✅ 成功导出 {len(samples)} 个样本到 {output_path}")
        print("📋 样本列表：")
        for s in samples:
            print(f"  - id={s['id']}, scene_id={s['scene_id']}")


if __name__ == "__main__":
    asyncio.run(main())