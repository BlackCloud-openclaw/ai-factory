#!/usr/bin/env python
"""清理测试数据 - 安全版本，检查表结构和列是否存在

支持因果引擎新增的表：
- predicates
- projection_applied
- projection_health
- chapter_budget
- affordance_usage
- projection_dead_letters
- event_embeddings
- narrative_causality
"""

import asyncio
import asyncpg
import sys
import os
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import config

DEFAULT_NOVEL_ID = "simple_long_novel_001"


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


async def table_exists(conn, table_name: str) -> bool:
    """检查表是否存在"""
    try:
        row = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = $1
            )
        """, table_name)
        return row
    except Exception:
        return False


async def clean_database(novel_id: str):
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
                    novel_id
                )
                print(f"✅ Cleaned narrative_events: {result}")
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
                    novel_id
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
                    novel_id
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
                    novel_id
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
                    novel_id
                )
                print(f"✅ Cleaned chapters: {result}")
            else:
                print(f"⚠️ chapters has no novel_id column, columns: {columns}")
        except Exception as e:
            print(f"⚠️ Could not clean chapters: {e}")
        
        # 6. 清理 writing_progress（使用 project_id 字段）
        try:
            result = await conn.execute(
                "DELETE FROM writing_progress WHERE project_id = $1",
                novel_id
            )
            print(f"✅ Cleaned writing_progress: {result}")
        except Exception as e:
            print(f"⚠️ Could not clean writing_progress: {e}")
        
        # 7. 清理 scene_execution_units
        try:
            result = await conn.execute(
                "DELETE FROM scene_execution_units WHERE novel_id = $1",
                novel_id
            )
            print(f"✅ Cleaned scene_execution_units: {result}")
        except Exception as e:
            print(f"⚠️ Could not clean scene_execution_units: {e}")
        
        # 8. 清理 resume_tasks
        try:
            result = await conn.execute(
                "DELETE FROM resume_tasks WHERE novel_id = $1",
                novel_id
            )
            print(f"✅ Cleaned resume_tasks: {result}")
        except Exception as e:
            print(f"⚠️ Could not clean resume_tasks: {e}")
        
        # ========== 新增：因果引擎相关表 ==========
        
        # 9. 清理 predicates
        try:
            result = await conn.execute(
                "DELETE FROM predicates WHERE novel_id = $1",
                novel_id
            )
            print(f"✅ Cleaned predicates: {result}")
        except Exception as e:
            print(f"⚠️ Could not clean predicates: {e}")
        
        # 10. 清理 projection_applied
        try:
            result = await conn.execute(
                "DELETE FROM projection_applied WHERE novel_id = $1",
                novel_id
            )
            print(f"✅ Cleaned projection_applied: {result}")
        except Exception as e:
            print(f"⚠️ Could not clean projection_applied: {e}")
        
        # 11. 清理 projection_health
        try:
            result = await conn.execute(
                "DELETE FROM projection_health WHERE novel_id = $1",
                novel_id
            )
            print(f"✅ Cleaned projection_health: {result}")
        except Exception as e:
            print(f"⚠️ Could not clean projection_health: {e}")
        
        # 12. 清理 chapter_budget
        try:
            result = await conn.execute(
                "DELETE FROM chapter_budget WHERE novel_id = $1",
                novel_id
            )
            print(f"✅ Cleaned chapter_budget: {result}")
        except Exception as e:
            print(f"⚠️ Could not clean chapter_budget: {e}")
        
        # 13. 清理 affordance_usage
        try:
            result = await conn.execute(
                "DELETE FROM affordance_usage WHERE novel_id = $1",
                novel_id
            )
            print(f"✅ Cleaned affordance_usage: {result}")
        except Exception as e:
            print(f"⚠️ Could not clean affordance_usage: {e}")
        
        # 14. 清理 projection_dead_letters
        try:
            result = await conn.execute(
                "DELETE FROM projection_dead_letters WHERE novel_id = $1",
                novel_id
            )
            print(f"✅ Cleaned projection_dead_letters: {result}")
        except Exception as e:
            print(f"⚠️ Could not clean projection_dead_letters: {e}")
        
        # 15. 清理 event_embeddings（检查表是否存在）
        if await table_exists(conn, 'event_embeddings'):
            try:
                await conn.execute("""
                    DELETE FROM event_embeddings 
                    WHERE event_id IN (SELECT id FROM narrative_events WHERE novel_id = $1)
                """, novel_id)
                print(f"✅ Cleaned event_embeddings")
            except Exception as e:
                print(f"⚠️ Could not clean event_embeddings: {e}")
        else:
            print(f"ℹ️ event_embeddings table does not exist, skipping")
        
        # 16. 清理 narrative_causality（检查表是否存在）
        if await table_exists(conn, 'narrative_causality'):
            try:
                await conn.execute("""
                    DELETE FROM narrative_causality 
                    WHERE cause_event_id IN (SELECT id FROM narrative_events WHERE novel_id = $1)
                       OR effect_event_id IN (SELECT id FROM narrative_events WHERE novel_id = $1)
                """, novel_id)
                print(f"✅ Cleaned narrative_causality")
            except Exception as e:
                print(f"⚠️ Could not clean narrative_causality: {e}")
        else:
            print(f"ℹ️ narrative_causality table does not exist, skipping")
        
        # ============================================
        
        # 17. 重置 novels 表记录
        try:
            columns = await get_table_columns(conn, 'novels')
            if 'novel_id' in columns:
                exists = await conn.fetchval(
                    "SELECT EXISTS (SELECT 1 FROM novels WHERE novel_id = $1)",
                    novel_id
                )
                if exists:
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
                    """, novel_id)
                    print(f"✅ Reset novels record: {result}")
                else:
                    print(f"ℹ️ No novels record found for {novel_id}")
            else:
                print(f"⚠️ novels table columns: {columns}")
        except Exception as e:
            print(f"⚠️ Could not clean novels: {e}")
        
        # 18. 可选：列出当前 novels 表中的所有记录（调试）
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


def clean_files(novel_id: str):
    """删除生成的小说文件"""
    novel_dir = Path(f"data/novels/{novel_id}")
    if novel_dir.exists():
        shutil.rmtree(novel_dir)
        print(f"✅ Deleted {novel_dir}")
    else:
        print(f"ℹ️ {novel_dir} does not exist")


async def main():
    parser = argparse.ArgumentParser(description="Clean test data for a specific novel")
    parser.add_argument("novel_id", nargs="?", default=DEFAULT_NOVEL_ID, 
                        help=f"Novel ID to clean (default: {DEFAULT_NOVEL_ID})")
    args = parser.parse_args()
    
    novel_id = args.novel_id
    
    print(f"Cleaning test data for novel: {novel_id}")
    print("Database DSN:", config.postgres_dsn.replace(config.postgres_password, "***"))
    print("-" * 50)
    
    await clean_database(novel_id)
    clean_files(novel_id)
    
    print("-" * 50)
    print("✅ Clean completed! Test environment is ready.")


if __name__ == "__main__":
    asyncio.run(main())