"""LLM connection pool with concurrent request management."""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from src.common.logging import setup_logging

logger = setup_logging("execution.llm_pool")


@dataclass
class PoolStats:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timed_out_requests: int = 0
    rejected_requests: int = 0
    active_requests: int = 0


class LLMPool:
    def __init__(self, max_concurrent: int = 4, max_queue_size: int = 20, timeout: int = 120):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.timeout = timeout

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue_size = asyncio.Semaphore(max_queue_size)
        self._lock = asyncio.Lock()
        self._stats = PoolStats()
        self._started_at = time.time()
        self._running = True

    async def execute(self, func: Callable, *args: Any, model: Optional[str] = None, **kwargs: Any) -> Any:
        if not self._running:
            self._stats.rejected_requests += 1
            raise RuntimeError("LLM pool is shut down")

        await self._queue_size.acquire()

        try:
            async with self._semaphore:
                async with self._lock:
                    self._stats.total_requests += 1
                    self._stats.active_requests += 1

                try:
                    if model:
                        kwargs["model"] = model

                    result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout)

                    async with self._lock:
                        self._stats.successful_requests += 1
                    return result

                except asyncio.TimeoutError:
                    async with self._lock:
                        self._stats.timed_out_requests += 1
                    raise

                except Exception:
                    async with self._lock:
                        self._stats.failed_requests += 1
                    raise

        finally:
            self._queue_size.release()
            async with self._lock:
                self._stats.active_requests -= 1

    def get_stats(self) -> Dict[str, Any]:
        uptime = time.time() - self._started_at
        return {
            "max_concurrent": self.max_concurrent,
            "max_queue_size": self.max_queue_size,
            "timeout": self.timeout,
            "uptime_seconds": round(uptime, 1),
            "total_requests": self._stats.total_requests,
            "successful_requests": self._stats.successful_requests,
            "failed_requests": self._stats.failed_requests,
            "timed_out_requests": self._stats.timed_out_requests,
            "rejected_requests": self._stats.rejected_requests,
            "active_requests": self._stats.active_requests,
            "success_rate": round(self._stats.successful_requests / max(self._stats.total_requests, 1) * 100, 1) if self._stats.total_requests else 0.0,
        }

    async def shutdown(self) -> None:
        self._running = False
        logger.info("LLM pool shutting down")

    def reset_stats(self) -> None:
        self._stats = PoolStats()
        self._started_at = time.time()