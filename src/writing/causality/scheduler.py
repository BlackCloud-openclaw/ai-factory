"""投影调度器 - 确保同一 novel 的投影串行执行"""
import asyncio
import time
import logging
from typing import Dict

from src.db import get_db_pool
from src.config import config
from .projection_store import ProjectionStore
from .delta import PredicateDelta

logger = logging.getLogger(__name__)


class ProjectionScheduler:
    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = {}
        self._store = ProjectionStore()

    async def schedule(self, novel_id: str, delta: PredicateDelta):
        lock = self._get_lock(novel_id)
        async with lock:
            start = time.perf_counter()
            
            # 幂等检查
            delta_id = self._store._make_delta_id(delta.novel_id, delta.event_id, delta.projection_version)
            if await self._store._is_applied(delta_id):
                logger.debug(f"Delta {delta_id} already applied, skipping")
                return
            
            success = await self._store.apply_delta(delta)
            elapsed = time.perf_counter() - start
            
            # 记录投影延迟并更新投影健康状态（如果启用）
            if config.enable_projection_metrics:
                async with self._store.pool.acquire() as conn:
                    # 更新投影健康表：记录最后投影的事件ID，重置滞后计数
                    # 使用 INSERT ... ON CONFLICT 确保行存在
                    await conn.execute("""
                        INSERT INTO projection_health (novel_id, last_projected_event_id, projection_lag_events, updated_at)
                        VALUES ($1, $2, 0, NOW())
                        ON CONFLICT (novel_id) DO UPDATE SET
                            last_projected_event_id = EXCLUDED.last_projected_event_id,
                            projection_lag_events = 0,
                            updated_at = NOW()
                    """, novel_id, delta.event_id)
                    
                    # 记录投影延迟指标
                    await conn.execute("""
                        INSERT INTO projection_metrics (novel_id, event_id, latency_seconds, created_at)
                        VALUES ($1, $2, $3, NOW())
                    """, novel_id, delta.event_id, elapsed)
            
            if not success:
                logger.warning(f"Projection for event {delta.event_id} took {elapsed:.3f}s but failed")
            else:
                logger.info(f"Projection for event {delta.event_id} applied in {elapsed:.3f}s")

    def _get_lock(self, novel_id: str) -> asyncio.Lock:
        if novel_id not in self._locks:
            self._locks[novel_id] = asyncio.Lock()
        return self._locks[novel_id]