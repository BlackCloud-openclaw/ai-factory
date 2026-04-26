# src/execution/llm_router_pool.py
import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

from src.common.logging import setup_logging

logger = setup_logging("execution.llm_router_pool")


@dataclass
class ModelSlot:
    """模型槽位：维护一个模型的并发限制和访问信息"""
    name: str
    max_concurrent: int = 2
    _semaphore: asyncio.Semaphore = field(init=False)
    last_used: float = field(default_factory=time.time)
    active_tasks: int = 0

    def __post_init__(self):
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

    async def acquire(self):
        await self._semaphore.acquire()
        self.active_tasks += 1
        self.last_used = time.time()

    def release(self):
        self._semaphore.release()
        self.active_tasks -= 1


class LLMRouterPool:
    """
    模型路由池：支持自动选模型、并发控制、排队、fallback。
    不包含模型卸载功能（llama.cpp 通常不支持）。
    """

    def __init__(self, default_timeout: int = 120):
        self.default_timeout = default_timeout
        self.model_slots: Dict[str, ModelSlot] = {}
        self._lock = asyncio.Lock()

        # 大模型集合，用于额外的互斥（防止两个大模型同时占用大量内存）
        self.large_models = {
            "Qwen3.5-122B-A10B-Q4_K_M",
            "DeepSeek-R1-Distill-Llama-70B-Q5_K_M",
        }
        self.large_model_semaphore = asyncio.Semaphore(1)

    def register_model(self, name: str, max_concurrent: int = 2):
        """注册模型槽位（如果还没注册）"""
        if name not in self.model_slots:
            self.model_slots[name] = ModelSlot(name, max_concurrent)

    async def call(
        self,
        model_name: str,
        func: Callable,
        *args,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        调用指定模型，受模型槽位和并发控制，支持超时。
        注意：func 必须接受 model_name 作为第一个参数，其余参数通过 args/kwargs 传递。
        """
        # 确保槽位存在
        if model_name not in self.model_slots:
            async with self._lock:
                self.register_model(model_name)

        slot = self.model_slots[model_name]

        # 大模型额外限流（互斥）
        if model_name in self.large_models:
            async with self.large_model_semaphore:
                await slot.acquire()
                try:
                    return await asyncio.wait_for(
                        func(model_name, *args, **kwargs),
                        timeout=timeout or self.default_timeout
                    )
                finally:
                    slot.release()
        else:
            await slot.acquire()
            try:
                return await asyncio.wait_for(
                    func(model_name, *args, **kwargs),
                        timeout=timeout or self.default_timeout
                    )
            finally:
                slot.release()

    async def call_with_fallback(
        self,
        model_candidates: List[str],
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        按顺序尝试候选模型，直到成功或全部失败。
        """
        last_exception = None
        for model in model_candidates:
            try:
                return await self.call(model, func, *args, **kwargs)
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}, trying next candidate")
                last_exception = e
        raise last_exception or RuntimeError("All candidate models failed")

    async def get_stats(self) -> Dict[str, Any]:
        """获取模型池统计信息"""
        stats = {}
        async with self._lock:
            for name, slot in self.model_slots.items():
                stats[name] = {
                    "active_tasks": slot.active_tasks,
                    "max_concurrent": slot.max_concurrent,
                    "last_used": slot.last_used,
                }
        return stats


# 全局单例
_pool: Optional[LLMRouterPool] = None

def get_llm_router_pool() -> LLMRouterPool:
    global _pool
    if _pool is None:
        _pool = LLMRouterPool()
        # 预注册常用模型及其并发限制（根据模型大小设置不同并发）
        _pool.register_model("Qwen3-Coder-30B-A3B-Instruct-Q5_K_M", max_concurrent=2)
        _pool.register_model("Qwen3.6-35B-A3B-UD-Q5_K_M", max_concurrent=2)
        _pool.register_model("DeepSeek-R1-Distill-Qwen-32B-Q5_K_M", max_concurrent=1)
        _pool.register_model("DeepSeek-R1-Distill-Llama-70B-Q5_K_M", max_concurrent=1)
        _pool.register_model("Qwen3.5-122B-A10B-Q4_K_M", max_concurrent=1)
        _pool.register_model("Qwen3.5-9B-Q4_K_M", max_concurrent=4)
        _pool.register_model("MiniMax-M2.7-UD-Q3_K_XL", max_concurrent=4)
    return _pool