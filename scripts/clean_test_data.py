#!/usr/bin/env python
"""
清理测试数据 - 完整版

清理指定小说的所有相关数据，包括：
- 事件溯源：narrative_events, world_snapshots, predicates, projection_*
- 章节内容：chapters, chapter_summaries, narrative_versions
- 进度状态：writing_progress, scene_execution_units, resume_tasks
- 叙事控制：loop_store, narrative_projection_snapshots
- 审计日志：state_audit, shadow_rewrite_log
- 知识检索：materials
- 主记录：novels

支持 --dry-run 预览，--yes 跳过确认。
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


# ============================================================
# 表清单（按 novel_id 清理）
# ============================================================
TABLES_WITH_NOVEL_ID = [
    # 事件溯源核心
    "narrative_events",
    "world_snapshots",
    "predicates",
    "projection_applied",
    "projection_health",
    "projection_dead_letters",
    "projection_metrics",
    "compressed_states",

    # 章节内容
    "chapters",
    "chapter_summaries",
    "narrative_versions",

    # 进度状态
    "writing_progress",     # 使用 project_id
    "scene_execution_units",

    # 叙事控制
    "loop_store",
    "narrative_projection_snapshots",

    # 审计
    "state_audit",

    # 知识检索
    "materials",

    # 主记录
    "novels",
]


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
        print(f"  ⚠️ Could not get columns for {table_name}: {e}")
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


async def count_rows(conn, table_name: str, id_col: str, novel_id: str) -> int:
    """统计待删除行数"""
    try:
        if not await table_exists(conn, table_name):
            return -1
        columns = await get_table_columns(conn, table_name)
        if id_col not in columns:
            return -1
        result = await conn.fetchval(
            f"SELECT COUNT(*) FROM {table_name} WHERE {id_col} = $1",
            novel_id
        )
        return result or 0
    except Exception:
        return -1


async def clean_database(novel_id: str, dry_run: bool = False):
    """清理数据库中的相关记录"""
    print("Connecting to database...")
    conn = await asyncpg.connect(config.postgres_dsn)

    stats = {}

    try:
        # ---------- 1. 按 novel_id 清理 ----------
        print("\n" + "=" * 60)
        print("按 novel_id 清理")
        print("=" * 60)

        for table in TABLES_WITH_NOVEL_ID:
            if not await table_exists(conn, table):
                print(f"ℹ️ {table}: table does not exist, skipping")
                stats[table] = -1
                continue

            # writing_progress 使用 project_id
            id_col = "project_id" if table == "writing_progress" else "novel_id"

            columns = await get_table_columns(conn, table)
            if id_col not in columns:
                print(f"⚠️ {table}: no {id_col} column (columns={columns[:5]}...), skipping")
                stats[table] = -1
                continue

            count = await count_rows(conn, table, id_col, novel_id)

            if dry_run:
                print(f"🔍 [DRY-RUN] {table}: would delete {count} rows")
                stats[table] = count
                continue

            try:
                result = await conn.execute(
                    f"DELETE FROM {table} WHERE {id_col} = $1",
                    novel_id
                )
                print(f"✅ {table}: {result}")
                stats[table] = count
            except Exception as e:
                print(f"⚠️ {table}: failed - {e}")
                stats[table] = -1

        # ---------- 2. 清理 shadow_rewrite_log（按 scene_id 前缀） ----------
        print("\n" + "=" * 60)
        print("清理 shadow_rewrite_log（按 scene_id 前缀）")
        print("=" * 60)

        if await table_exists(conn, 'shadow_rewrite_log'):
            # scene_id 格式：scene_5_47_0，与 novel_id 无直接关联
            # 我们假设当前只有一个 novel 在跑，直接清空
            try:
                count = await conn.fetchval("SELECT COUNT(*) FROM shadow_rewrite_log")
                if dry_run:
                    print(f"🔍 [DRY-RUN] shadow_rewrite_log: would delete {count} rows")
                else:
                    result = await conn.execute("DELETE FROM shadow_rewrite_log")
                    print(f"✅ shadow_rewrite_log: {result}")
                stats["shadow_rewrite_log"] = count
            except Exception as e:
                print(f"⚠️ shadow_rewrite_log: failed - {e}")
                stats["shadow_rewrite_log"] = -1
        else:
            print("ℹ️ shadow_rewrite_log: table does not exist, skipping")

        # ---------- 3. 清理 task_jobs（可选） ----------
        print("\n" + "=" * 60)
        print("清理 task_jobs")
        print("=" * 60)

        if await table_exists(conn, 'task_jobs'):
            try:
                # task_jobs 通过 description 或 subtask_id 关联，无 novel_id
                # 保守起见：不清理（可能影响其他任务）
                print("ℹ️ task_jobs: 无 novel_id 关联，跳过（避免误删其他任务）")
                stats["task_jobs"] = -1
            except Exception as e:
                print(f"⚠️ task_jobs: failed - {e}")
        else:
            print("ℹ️ task_jobs: table does not exist, skipping")

        # ---------- 4. 清理 event_embeddings（通过 narrative_events 关联） ----------
        print("\n" + "=" * 60)
        print("清理 event_embeddings（通过事件关联）")
        print("=" * 60)

        if await table_exists(conn, 'event_embeddings'):
            try:
                count = await conn.fetchval("""
                    SELECT COUNT(*) FROM event_embeddings 
                    WHERE event_id IN (
                        SELECT id FROM narrative_events WHERE novel_id = $1
                    )
                """, novel_id)
                if dry_run:
                    print(f"🔍 [DRY-RUN] event_embeddings: would delete {count} rows")
                else:
                    await conn.execute("""
                        DELETE FROM event_embeddings 
                        WHERE event_id IN (
                            SELECT id FROM narrative_events WHERE novel_id = $1
                        )
                    """, novel_id)
                    print(f"✅ event_embeddings: {count} rows deleted")
                stats["event_embeddings"] = count
            except Exception as e:
                print(f"⚠️ event_embeddings: failed - {e}")
        else:
            print("ℹ️ event_embeddings: table does not exist, skipping")

        # ---------- 5. 清理 narrative_causality（通过事件关联） ----------
        print("\n" + "=" * 60)
        print("清理 narrative_causality（通过事件关联）")
        print("=" * 60)

        if await table_exists(conn, 'narrative_causality'):
            try:
                count = await conn.fetchval("""
                    SELECT COUNT(*) FROM narrative_causality 
                    WHERE cause_event_id IN (SELECT id FROM narrative_events WHERE novel_id = $1)
                       OR effect_event_id IN (SELECT id FROM narrative_events WHERE novel_id = $1)
                """, novel_id)
                if dry_run:
                    print(f"🔍 [DRY-RUN] narrative_causality: would delete {count} rows")
                else:
                    await conn.execute("""
                        DELETE FROM narrative_causality 
                        WHERE cause_event_id IN (SELECT id FROM narrative_events WHERE novel_id = $1)
                           OR effect_event_id IN (SELECT id FROM narrative_events WHERE novel_id = $1)
                    """, novel_id)
                    print(f"✅ narrative_causality: {count} rows deleted")
                stats["narrative_causality"] = count
            except Exception as e:
                print(f"⚠️ narrative_causality: failed - {e}")
        else:
            print("ℹ️ narrative_causality: table does not exist, skipping")

        # ---------- 6. 汇总 ----------
        print("\n" + "=" * 60)
        print("汇总")
        print("=" * 60)
        total = sum(v for v in stats.values() if v > 0)
        print(f"总计清理: {total} 行")
        print(f"涉及表数: {len([v for v in stats.values() if v >= 0])}")

    finally:
        await conn.close()

    return stats


def clean_files(novel_id: str, dry_run: bool = False):
    """删除生成的小说文件"""
    novel_dir = Path(f"data/novels/{novel_id}")
    if novel_dir.exists():
        if dry_run:
            print(f"🔍 [DRY-RUN] Would delete directory: {novel_dir}")
            # 列出目录内容大小
            total_size = sum(
                f.stat().st_size for f in novel_dir.rglob('*') if f.is_file()
            )
            file_count = sum(1 for f in novel_dir.rglob('*') if f.is_file())
            print(f"    包含 {file_count} 个文件，共 {total_size / 1024:.1f} KB")
        else:
            shutil.rmtree(novel_dir)
            print(f"✅ Deleted {novel_dir}")
    else:
        print(f"ℹ️ {novel_dir} does not exist")


def clean_debug_files(dry_run: bool = False):
    """清理调试文件"""
    debug_dir = Path("logs/debug")
    if debug_dir.exists():
        files = list(debug_dir.glob("writer_parse_failed_*.json"))
        if dry_run:
            print(f"🔍 [DRY-RUN] Would delete {len(files)} debug files from {debug_dir}")
        else:
            for f in files:
                f.unlink()
            print(f"✅ Deleted {len(files)} debug files from {debug_dir}")
    else:
        print(f"ℹ️ {debug_dir} does not exist")


async def main():
    parser = argparse.ArgumentParser(
        description="Clean test data for a specific novel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/clean_test_data.py                    # 清理默认小说
  python scripts/clean_test_data.py --dry-run          # 预览
  python scripts/clean_test_data.py --yes              # 跳过确认
  python scripts/clean_test_data.py my_novel_002       # 指定小说
"""
    )
    parser.add_argument(
        "novel_id",
        nargs="?",
        default=DEFAULT_NOVEL_ID,
        help=f"Novel ID to clean (default: {DEFAULT_NOVEL_ID})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览，不实际删除"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="跳过确认提示"
    )
    parser.add_argument(
        "--no-files",
        action="store_true",
        help="仅清理数据库，不删除文件"
    )
    args = parser.parse_args()

    novel_id = args.novel_id

    print("=" * 60)
    print(f"清理测试数据")
    print("=" * 60)
    print(f"Novel ID: {novel_id}")
    print(f"模式: {'DRY-RUN (预览)' if args.dry_run else '实际删除'}")
    print(f"DSN: {config.postgres_dsn.replace(config.postgres_password, '***')}")
    print("-" * 60)

    # 确认
    if not args.dry_run and not args.yes:
        confirm = input(f"\n⚠️ 即将删除 {novel_id} 的所有数据，确认？[y/N]: ")
        if confirm.lower() not in ('y', 'yes'):
            print("已取消")
            return

    # 数据库清理
    await clean_database(novel_id, dry_run=args.dry_run)

    # 文件清理
    if not args.no_files:
        print("\n" + "=" * 60)
        print("清理文件")
        print("=" * 60)
        clean_files(novel_id, dry_run=args.dry_run)

        print("\n" + "=" * 60)
        print("清理调试文件")
        print("=" * 60)
        clean_debug_files(dry_run=args.dry_run)

    print("\n" + "=" * 60)
    if args.dry_run:
        print("✅ DRY-RUN 完成，未做任何修改")
        print("   如需实际删除，去掉 --dry-run 参数")
    else:
        print("✅ 清理完成！")
        print(f"   可以重新运行: python scripts/simple_long_novel.py")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())