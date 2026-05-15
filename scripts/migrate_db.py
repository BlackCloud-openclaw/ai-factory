#!/usr/bin/env python
"""数据库迁移：为 novels 表添加 scene_plan_list 列（如果不存在），并可选重命名 current_scene 列。"""

import asyncio
import asyncpg
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config

async def migrate():
    dsn = config.postgres_dsn
    conn = await asyncpg.connect(dsn)

    # 1. 检查并添加 scene_plan_list 列
    col_check = await conn.fetchval("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name='novels' AND column_name='scene_plan_list'
        )
    """)
    if not col_check:
        await conn.execute("ALTER TABLE novels ADD COLUMN scene_plan_list JSONB")
        print("✅ Added scene_plan_list column to novels")
    else:
        print("ℹ️ scene_plan_list column already exists")

    # 2. 可选：重命名 current_scene 为 current_scene_index（如果需要）
    # 注意：如果代码中已改用 current_scene_index 访问该列，则建议重命名；否则可跳过。
    rename = True  # 设为 False 则跳过重命名
    if rename:
        col_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='novels' AND column_name='current_scene'
            )
        """)
        new_name_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='novels' AND column_name='current_scene_index'
            )
        """)
        if col_exists and not new_name_exists:
            await conn.execute("ALTER TABLE novels RENAME COLUMN current_scene TO current_scene_index")
            print("✅ Renamed current_scene to current_scene_index")
        elif new_name_exists:
            print("ℹ️ current_scene_index already exists, skipping rename")
        else:
            print("⚠️ current_scene column not found, cannot rename")

    # 3. 确认所有列
    columns = await conn.fetch("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name='novels' ORDER BY ordinal_position
    """)
    print("\n📋 novels table columns after migration:")
    for col in columns:
        print(f"   {col['column_name']}")

    await conn.close()

if __name__ == "__main__":
    asyncio.run(migrate())