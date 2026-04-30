# src/api/scheduler.py
import asyncio
import logging
from typing import Callable, Any

logger = logging.getLogger("api.scheduler")

class PriorityScheduler:
    def __init__(self, worker_count: int = 3, max_retries: int = 10, task_timeout: int = 1800):
        self._queue = asyncio.PriorityQueue()
        self._counter = 0
        self._workers = []
        self._running = True
        self.max_retries = max_retries          # 最大重试次数
        self.task_timeout = task_timeout        # 整体超时 30 分钟
        self._submit_lock = asyncio.Lock()
        for _ in range(worker_count):
            self._workers.append(asyncio.create_task(self._worker()))

    async def submit(self, priority: int, coro: Callable, *args, **kwargs) -> Any:
        future = asyncio.Future()
        async with self._submit_lock:
            self._counter += 1
            await self._queue.put((priority, self._counter, future, coro, args, kwargs, 0))
        return await future

    async def _worker(self):
        while self._running:
            try:
                item = await self._queue.get()
                if not isinstance(item, tuple) or len(item) != 7:
                    logger.error(f"Invalid queue item: {item}")
                    continue
                priority, _, future, coro, args, kwargs, retry_count = item
                if future is None or coro is None:
                    logger.error(f"Future or coro is None in item: {item}")
                    continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker failed to get task: {e}", exc_info=True)
                continue

            try:
                result = await asyncio.wait_for(coro(*args, **kwargs), timeout=self.task_timeout)
                if not future.done():
                    future.set_result(result)
                continue   # 成功，跳过重试逻辑
            except asyncio.TimeoutError as e:
                error_msg = f"Task timed out after {self.task_timeout}s"
                logger.error(error_msg)
                last_exception = e
            except Exception as e:
                error_msg = str(e)
                last_exception = e

            # 重试逻辑（指数退避，保持原优先级）
            if retry_count >= self.max_retries:
                if not future.done():
                    future.set_exception(RuntimeError(f"Task exceeded max retries ({self.max_retries})"))
                continue

            # 指数退避，最大 30 秒
            delay = min(2 ** (retry_count + 1), 30)
            logger.warning(f"Task failed (retry {retry_count+1}/{self.max_retries}), delay {delay}s: {error_msg[:200]}")
            await asyncio.sleep(delay)

            # 保持原优先级（不增加 priority 数字）
            async with self._submit_lock:
                self._counter += 1
                await self._queue.put((priority, self._counter, future, coro, args, kwargs, retry_count + 1))

    async def shutdown(self):
        self._running = False
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

_scheduler: PriorityScheduler = None

def get_scheduler() -> PriorityScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = PriorityScheduler(worker_count=3, max_retries=10, task_timeout=1800)
    return _scheduler