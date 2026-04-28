# src/api/scheduler.py
import asyncio
from typing import Callable, Any

class PriorityScheduler:
    def __init__(self, worker_count: int = 3, max_retries: int = 20):
        self._queue = asyncio.PriorityQueue()
        self._counter = 0  # 确保元组排序稳定
        self._workers = []
        self._running = True
        self.max_retries = max_retries
        for _ in range(worker_count):
            self._workers.append(asyncio.create_task(self._worker()))

    async def submit(self, priority: int, coro: Callable, *args, **kwargs) -> Any:
        future = asyncio.Future()
        self._counter += 1
        await self._queue.put((priority, self._counter, future, coro, args, kwargs, 0))
        return await future

    async def _worker(self):
        while self._running:
            try:
                priority, _, future, coro, args, kwargs, retry_count = await self._queue.get()
            except Exception:
                continue

            try:
                result = await coro(*args, **kwargs)
                if not future.done():
                    future.set_result(result)
            except Exception as e:
                msg = str(e)
                if "Not enough memory" in msg or "Memory" in msg:
                    if retry_count >= self.max_retries:
                        if not future.done():
                            future.set_exception(RuntimeError(f"Task exceeded max retries ({self.max_retries}) due to memory"))
                        continue

                    # 退避延迟
                    delay = min(2 ** retry_count, 5)
                    await asyncio.sleep(delay)

                    # 降低优先级（+1），避免饿死其他任务
                    new_priority = priority + 1
                    self._counter += 1
                    await self._queue.put((new_priority, self._counter, future, coro, args, kwargs, retry_count + 1))
                else:
                    if not future.done():
                        future.set_exception(e)

    async def shutdown(self):
        self._running = False
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)


_scheduler: PriorityScheduler = None

def get_scheduler() -> PriorityScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = PriorityScheduler(worker_count=3, max_retries=20)
    return _scheduler