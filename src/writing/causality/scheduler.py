"""投影调度器 - 确保同一 novel 的投影串行执行"""
import asyncio
import logging
from typing import Dict

from src.db import get_db_pool
from .projection_store import ProjectionStore
from .delta import PredicateDelta

logger = logging.getLogger(__name__)


class ProjectionScheduler:
    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = {}
        self._store = ProjectionStore()

    async def schedule(self, novel_id: str, delta: PredicateDelta):
        """串行执行投影任务"""
        lock = self._get_lock(novel_id)
        async with lock:
            # 先检查是否已经应用（幂等）
            if await self._store._is_applied(
                ProjectionStore._make_delta_id(delta.novel_id, delta.event_id, delta.projection_version)
            ):
                logger.debug(f"Delta for event {delta.event_id} already applied, skipping")
                return
            # 执行应用
            success = await self._store.apply_delta(delta)
            if not success:
                logger.warning(f"Projection for event {delta.event_id} failed, moved to dead letter")
            else:
                logger.info(f"Projection for event {delta.event_id} applied successfully")

    def _get_lock(self, novel_id: str) -> asyncio.Lock:
        if novel_id not in self._locks:
            self._locks[novel_id] = asyncio.Lock()
        return self._locks[novel_id]