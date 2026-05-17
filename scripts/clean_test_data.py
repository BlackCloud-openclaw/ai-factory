#!/usr/bin/env python
"""清理测试数据 - 安全版本，检查表结构和列是否存在"""

import asyncio
import asyncpg
import sys
import os
import shutil
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import config

NOVEL_ID = "simple_long_novel_001"

async def get_table_columns(conn, table_name):
    """获取表的列名列表"""
    try:
        rows = await conn.fetch("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = $1
            ORDER BY ordinal_position
        """, table_name)
        return [row['column_name'] for row in rows]
    except Exception as e:
        print(f"  Warning: Could not get columns for {table_name}: {e}")
        return []

async def clean_database():
    """清理数据库中的相关记录（安全版本）"""
    print("Connecting to database...")
    conn = await asyncpg.connect(config.postgres_dsn)
    
    try:
        # 1. 清理 narrative_events
        try:
            columns = await get_table_columns(conn, 'narrative_events')
            if 'novel_id' in columns:
                result = await conn.execute(
                    "DELETE FROM narrative_events WHERE novel_id = $1",
                    NOVEL_ID
                )
                print(f"✅ Cleaned narrative_events: {result}")
            elif 'novel_id' in [c for c in columns if 'novel' in c.lower()]:
                # 尝试找到包含 novel 的列名
                novel_col = next((c for c in columns if 'novel' in c.lower()), None)
                if novel_col:
                    result = await conn.execute(
                        f"DELETE FROM narrative_events WHERE {novel_col} = $1",
                        NOVEL_ID
                    )
                    print(f"✅ Cleaned narrative_events (using {novel_col}): {result}")
            else:
                print(f"⚠️ narrative_events has no novel_id column, columns: {columns}")
        except Exception as e:
            print(f"⚠️ Could not clean narrative_events: {e}")
        
        # 2. 清理 world_snapshots
        try:
            columns = await get_table_columns(conn, 'world_snapshots')
            if 'novel_id' in columns:
                result = await conn.execute(
                    "DELETE FROM world_snapshots WHERE novel_id = $1",
                    NOVEL_ID
                )
                print(f"✅ Cleaned world_snapshots: {result}")
            else:
                print(f"⚠️ world_snapshots has no novel_id column, columns: {columns}")
        except Exception as e:
            print(f"⚠️ Could not clean world_snapshots: {e}")
        
        # 3. 清理 compressed_states
        try:
            columns = await get_table_columns(conn, 'compressed_states')
            if 'novel_id' in columns:
                result = await conn.execute(
                    "DELETE FROM compressed_states WHERE novel_id = $1",
                    NOVEL_ID
                )
                print(f"✅ Cleaned compressed_states: {result}")
            else:
                print(f"⚠️ compressed_states has no novel_id column, columns: {columns}")
        except Exception as e:
            print(f"⚠️ Could not clean compressed_states: {e}")
        
        # 4. 清理 chapter_summaries
        try:
            columns = await get_table_columns(conn, 'chapter_summaries')
            if 'novel_id' in columns:
                result = await conn.execute(
                    "DELETE FROM chapter_summaries WHERE novel_id = $1",
                    NOVEL_ID
                )
                print(f"✅ Cleaned chapter_summaries: {result}")
            else:
                print(f"⚠️ chapter_summaries has no novel_id column, columns: {columns}")
        except Exception as e:
            print(f"⚠️ Could not clean chapter_summaries: {e}")
        
        # 5. 清理 chapters
        try:
            columns = await get_table_columns(conn, 'chapters')
            if 'novel_id' in columns:
                result = await conn.execute(
                    "DELETE FROM chapters WHERE novel_id = $1",
                    NOVEL_ID
                )
                print(f"✅ Cleaned chapters: {result}")
            else:
                print(f"⚠️ chapters has no novel_id column, columns: {columns}")
        except Exception as e:
            print(f"⚠️ Could not clean chapters: {e}")
        
        # 6. 清理或重置 novels 表
        try:
            columns = await get_table_columns(conn, 'novels')
            if 'novel_id' in columns:
                # 检查记录是否存在
                exists = await conn.fetchval(
                    "SELECT EXISTS (SELECT 1 FROM novels WHERE novel_id = $1)",
                    NOVEL_ID
                )
                if exists:
                    # 重置记录而不是删除（保留 ID 但清空数据）
                    result = await conn.execute("""
                        UPDATE novels 
                        SET outline = NULL,
                            current_volume = 1,
                            current_chapter = 1,
                            current_scene_index = 0,
                            current_state = NULL,
                            last_sequence_id = 0,
                            revision = revision + 1,
                            updated_at = NOW()
                        WHERE novel_id = $1
                    """, NOVEL_ID)
                    print(f"✅ Reset novels record: {result}")
                else:
                    print(f"ℹ️ No novels record found for {NOVEL_ID}")
            else:
                print(f"⚠️ novels table columns: {columns}")
        except Exception as e:
            print(f"⚠️ Could not clean novels: {e}")
        
        # 7. 可选：列出当前 novels 表中的所有记录
        try:
            rows = await conn.fetch("SELECT novel_id, current_chapter, current_scene_index FROM novels LIMIT 5")
            if rows:
                print("\n📋 Current novels in database:")
                for row in rows:
                    print(f"   - {row['novel_id']}: chapter={row.get('current_chapter', 'N/A')}, scene={row.get('current_scene_index', 'N/A')}")
            else:
                print("\n📋 No novels found in database")
        except Exception as e:
            print(f"⚠️ Could not list novels: {e}")
        
    finally:
        await conn.close()

def clean_files():
    """删除生成的小说文件"""
    novel_dir = Path(f"data/novels/{NOVEL_ID}")
    if novel_dir.exists():
        shutil.rmtree(novel_dir)
        print(f"✅ Deleted {novel_dir}")
    else:
        print(f"ℹ️ {novel_dir} does not exist")

async def main():
    print(f"Cleaning test data for novel: {NOVEL_ID}")
    print("Database DSN:", config.postgres_dsn.replace(config.postgres_password, "***"))
    print("-" * 50)
    
    await clean_database()
    clean_files()
    
    print("-" * 50)
    print("✅ Clean completed! Test environment is ready.")

if __name__ == "__main__":
    asyncio.run(main())