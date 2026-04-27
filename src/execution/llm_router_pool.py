# src/execution/llm_router_pool.py
import asyncio
import time
import subprocess
import psutil
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

from src.common.logging import setup_logging

logger = setup_logging("execution.llm_router_pool")

# ========== 模型配置（容器名、端口、并发限制、内存估算GB） ==========
MODEL_CONFIG = {
    "Qwen3-Coder-30B-A3B-Instruct-Q5_K_M": {
        "container": "llamacpp-coder",
        "port": 8081,
        "max_concurrent": 2,
        "memory_gb": 35,
    },
    "Qwen3.6-27B-Q5_K_M": {
        "container": "llamacpp-writing",
        "port": 8082,
        "max_concurrent": 2,
        "memory_gb": 30,
    },
    "Qwen3.6-35B-A3B-UD-Q5_K_M": {
        "container": "llamacpp-default",
        "port": 8083,
        "max_concurrent": 2,
        "memory_gb": 40,
    },
    "Qwen3.5-9B-Q4_K_M": {
        "container": "llamacpp-fast",
        "port": 8084,
        "max_concurrent": 4,
        "memory_gb": 10,
    },
    "DeepSeek-R1-Distill-Qwen-32B-Q5_K_M": {
        "container": "llamacpp-reasoning",
        "port": 8085,
        "max_concurrent": 1,
        "memory_gb": 35,
    },
    "DeepSeek-R1-Distill-Llama-70B-Q5_K_M": {
        "container": "llamacpp-deepseek70b",
        "port": 8086,
        "max_concurrent": 1,
        "memory_gb": 70,
    },
    "Qwen3.5-122B-A10B-Q4_K_M": {
        "container": "llamacpp-qwen122b",
        "port": 8087,
        "max_concurrent": 1,
        "memory_gb": 80,
    },
    "MiniMax-M2.7-UD-Q3_K_XL": {
        "container": None,
        "port": 8081,
        "max_concurrent": 4,
        "memory_gb": 5,
    },
}

# 内存安全阈值（GB），启动新模型时至少保留的可用内存
MEMORY_SAFETY_MARGIN_GB = 10

# 大模型集合（用于全局互斥）
LARGE_MODELS = {
    "Qwen3.5-122B-A10B-Q4_K_M",
    "DeepSeek-R1-Distill-Llama-70B-Q5_K_M",
}


@dataclass
class ModelSlot:
    name: str
    max_concurrent: int = 2
    container_name: Optional[str] = None
    host_port: int = 8081
    memory_gb: int = 30
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
    def __init__(self, default_timeout: int = 120, idle_timeout: int = 600):
        self.default_timeout = default_timeout
        self.idle_timeout = idle_timeout
        self.model_slots: Dict[str, ModelSlot] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self.large_model_semaphore = asyncio.Semaphore(1)

        # 注册所有模型
        for name, cfg in MODEL_CONFIG.items():
            self.register_model(
                name,
                max_concurrent=cfg["max_concurrent"],
                container_name=cfg.get("container"),
                host_port=cfg["port"],
                memory_gb=cfg.get("memory_gb", 30),
            )

        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    def register_model(
        self,
        name: str,
        max_concurrent: int = 2,
        container_name: Optional[str] = None,
        host_port: int = 8081,
        memory_gb: int = 30,
    ):
        if name not in self.model_slots:
            self.model_slots[name] = ModelSlot(
                name=name,
                max_concurrent=max_concurrent,
                container_name=container_name,
                host_port=host_port,
                memory_gb=memory_gb,
            )
            logger.debug(f"Registered model {name} (container={container_name}, port={host_port}, memory={memory_gb}GB)")

    async def _is_container_running(self, container_name: str) -> bool:
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip() == "true"
        except Exception:
            return False

    async def _start_container(self, container_name: str):
        logger.info(f"Starting container {container_name}")
        subprocess.run(["docker", "start", container_name], capture_output=True, check=False)
        # 等待服务就绪（可轮询健康检查，此处简单等待）
        await asyncio.sleep(5)

    async def _stop_container(self, container_name: str):
        logger.info(f"Stopping idle container {container_name}")
        subprocess.run(["docker", "stop", container_name], capture_output=True, check=False)

    async def _ensure_memory_for_model(self, model_name: str) -> bool:
        """确保有足够内存启动模型，否则尝试驱逐其他空闲容器"""
        slot = self.model_slots.get(model_name)
        if not slot:
            return True  # 未配置内存估算，跳过检查

        required_gb = slot.memory_gb
        required_available = (required_gb + MEMORY_SAFETY_MARGIN_GB) * 1024 * 1024 * 1024

        mem = psutil.virtual_memory()
        if mem.available >= required_available:
            return True

        logger.warning(
            f"Low memory: available={mem.available // (1024**3)}GB, need at least {required_gb + MEMORY_SAFETY_MARGIN_GB}GB. "
            f"Attempting to evict idle containers."
        )

        # 收集运行中的其他模型容器（按最后使用时间排序）
        candidates = []
        for name, s in self.model_slots.items():
            if name == model_name:
                continue
            if s.container_name and await self._is_container_running(s.container_name):
                # 只驱逐没有活动任务且空闲时间较长的容器
                if s.active_tasks == 0:
                    candidates.append((s.last_used, name, s))

        candidates.sort(key=lambda x: x[0])  # 按 last_used 升序（最久未使用优先）

        for _, name, s in candidates:
            logger.info(f"Evicting idle container {s.container_name} for model {name} to free memory")
            await self._stop_container(s.container_name)
            await asyncio.sleep(1)  # 等待内存回收
            mem = psutil.virtual_memory()
            if mem.available >= required_available:
                return True

        logger.error(
            f"Insufficient memory even after eviction. Required: {required_gb + MEMORY_SAFETY_MARGIN_GB}GB, "
            f"available: {mem.available // (1024**3)}GB"
        )
        return False

    async def _ensure_model_ready(self, model_name: str):
        """确保模型容器正在运行，如果未运行则先检查内存并启动"""
        slot = self.model_slots.get(model_name)
        if not slot or not slot.container_name:
            return

        if not await self._is_container_running(slot.container_name):
            # 启动前检查内存
            if not await self._ensure_memory_for_model(model_name):
                raise RuntimeError(f"Not enough memory to start model {model_name}")
            await self._start_container(slot.container_name)

    async def _cleanup_loop(self):
        """定期停止空闲超时的容器"""
        while True:
            await asyncio.sleep(60)
            now = time.time()
            for name, slot in self.model_slots.items():
                if slot.active_tasks == 0 and (now - slot.last_used) > self.idle_timeout and slot.container_name:
                    if await self._is_container_running(slot.container_name):
                        await self._stop_container(slot.container_name)

    def get_base_url(self, model_name: str) -> str:
        slot = self.model_slots.get(model_name)
        if slot and slot.host_port:
            return f"http://localhost:{slot.host_port}"
        return "http://localhost:8081"

    async def call(
        self,
        model_name: str,
        func: Callable,
        *args,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Any:
        if model_name not in self.model_slots:
            async with self._lock:
                if model_name not in self.model_slots:
                    self.register_model(model_name)

        await self._ensure_model_ready(model_name)

        kwargs['base_url'] = self.get_base_url(model_name)

        slot = self.model_slots[model_name]

        if model_name in LARGE_MODELS:
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
        last_exception = None
        for model in model_candidates:
            try:
                return await self.call(model, func, *args, **kwargs)
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}, trying next candidate")
                last_exception = e
        raise last_exception or RuntimeError("All candidate models failed")

    async def get_stats(self) -> Dict[str, Any]:
        stats = {}
        async with self._lock:
            for name, slot in self.model_slots.items():
                stats[name] = {
                    "active_tasks": slot.active_tasks,
                    "max_concurrent": slot.max_concurrent,
                    "last_used": slot.last_used,
                    "container": slot.container_name,
                    "port": slot.host_port,
                    "memory_gb": slot.memory_gb,
                }
        return stats

    async def shutdown(self):
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass


_pool: Optional[LLMRouterPool] = None

def get_llm_router_pool() -> LLMRouterPool:
    global _pool
    if _pool is None:
        _pool = LLMRouterPool()
    return _pool