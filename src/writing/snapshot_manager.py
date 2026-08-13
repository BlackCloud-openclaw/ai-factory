"""
快照管理 - 支持旧快照自动迁移到 ID-key 格式

Phase 4C 变更：
- load_latest_snapshot() 加载旧快照时自动将 name-key 转换为 ID-key
- 迁移后立即保存新格式快照
- 保留旧快照（不删除），确保可回退
"""
import asyncpg
import json
import logging
from typing import Optional, Tuple
from datetime import datetime
from .world_state import WorldState
from .memory_hierarchy import CompressedState
from src.domain.identity import get_character_config, get_character_id_by_name

logger = logging.getLogger(__name__)


class SnapshotManager:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def save_snapshot(
        self,
        novel_id: str,
        world_state: WorldState,
        last_event_id: int,
        volume_num: int = None,
        chapter_num: int = None,
        compressed_state: Optional[CompressedState] = None,
        conn: Optional[asyncpg.Connection] = None,
    ) -> int:
        """
        保存快照，world_state 会被序列化为 ID-key 格式（由 model_dump 处理）
        """
        async with self.pool.acquire() as ac:
            async with ac.transaction():
                # 获取下一个 snapshot_id
                row = await ac.fetchrow(
                    "SELECT COALESCE(MAX(snapshot_id), 0) + 1 as next_id FROM world_snapshots WHERE novel_id = $1",
                    novel_id
                )
                snap_id = row["next_id"]

                # world_state.model_dump() 自动将 characters 键转换为 ID
                world_json = world_state.model_dump_json()
                compressed_json = compressed_state.model_dump_json() if compressed_state else None

                await ac.execute("""
                    INSERT INTO world_snapshots
                    (novel_id, snapshot_id, volume_num, chapter_num, last_event_id, world_state, compressed_state, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, novel_id, snap_id, volume_num, chapter_num, last_event_id,
                   world_json, compressed_json, datetime.now())

                logger.info(f"Saved snapshot {snap_id} for {novel_id} (event {last_event_id})")
                return snap_id

    async def load_latest_snapshot(self, novel_id: str) -> Tuple[Optional[WorldState], Optional[CompressedState], int]:
        """
        加载最新快照，并自动迁移旧格式（name-key → ID-key）
        迁移后立即保存新格式快照
        """
        config = get_character_config()

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT world_state, compressed_state, last_event_id
                FROM world_snapshots
                WHERE novel_id = $1
                ORDER BY snapshot_id DESC
                LIMIT 1
            """, novel_id)

            if not row:
                return None, None, 0

            # 加载原始数据
            data = json.loads(row["world_state"])
            world = WorldState.from_dict(data)

            # ---------- 迁移：检查并转换 characters 键 ----------
            needs_resave = False
            original_keys = list(world.characters.keys())
            normalized = {}

            for key, char_state in world.characters.items():
                # 1. 如果 key 已经是 ID（存在于配置中），直接保留
                if config.get_character(key) is not None:
                    normalized[key] = char_state
                    continue

                # 2. 如果 key 是名称，转换为 ID
                char_id = config.get_character_id_by_name(key)
                if char_id is not None:
                    normalized[char_id] = char_state
                    # 确保 CharacterState.id 已设置
                    if not hasattr(char_state, 'id') or char_state.id is None:
                        char_state.id = char_id
                    needs_resave = True
                    logger.info(f"Migrated character key from '{key}' to '{char_id}'")
                    continue

                # 3. 如果 char_state 本身有 id，使用它
                if hasattr(char_state, 'id') and char_state.id:
                    normalized[char_state.id] = char_state
                    needs_resave = True
                    continue

                # 4. 兜底：保留原键（但记录警告）
                normalized[key] = char_state
                logger.warning(f"Unknown character key '{key}' in snapshot, keeping as-is")

            # 如果发生了变化，更新 world.characters 并重新保存快照
            if needs_resave:
                world.characters = normalized
                logger.info(f"Re-saving migrated snapshot for {novel_id} (converted {len(original_keys)} keys)")

                # 解析 compressed_state（如果有）
                comp = None
                if row["compressed_state"]:
                    try:
                        comp = CompressedState.model_validate_json(row["compressed_state"])
                    except Exception as e:
                        logger.warning(f"Failed to parse compressed_state: {e}")

                # 立即保存新格式快照（使用同一个连接）
                await self.save_snapshot(
                    novel_id=novel_id,
                    world_state=world,
                    last_event_id=row["last_event_id"],
                    volume_num=None,
                    chapter_num=None,
                    compressed_state=comp,
                    conn=conn  # 复用连接，保证事务一致性
                )

                # 注意：旧快照保留，不删除（便于回退）
                logger.info(f"Migration complete for {novel_id}, old snapshot preserved")

            # ---------- 返回迁移后的状态 ----------
            comp = None
            if row["compressed_state"]:
                try:
                    comp = CompressedState.model_validate_json(row["compressed_state"])
                except Exception as e:
                    logger.warning(f"Failed to parse compressed_state: {e}")

            return world, comp, row["last_event_id"]

    async def prune_snapshots(self, novel_id: str, keep: int = 3):
        """保留最近 keep 个快照（注意：保留迁移后的新快照）"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                DELETE FROM world_snapshots
                WHERE novel_id = $1 AND snapshot_id NOT IN (
                    SELECT snapshot_id FROM world_snapshots
                    WHERE novel_id = $1
                    ORDER BY snapshot_id DESC
                    LIMIT $2
                )
            """, novel_id, keep)

    async def should_snapshot(self, novel_id: str, current_event_id: int, interval_events: int = 1000) -> bool:
        last_event_id = await self.get_last_snapshot_event_id(novel_id)
        return current_event_id - last_event_id >= interval_events

    async def get_last_snapshot_event_id(self, novel_id: str) -> int:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT last_event_id FROM world_snapshots
                WHERE novel_id = $1 ORDER BY snapshot_id DESC LIMIT 1
            """, novel_id)
            return row["last_event_id"] if row else 0