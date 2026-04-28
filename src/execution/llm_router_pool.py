# src/execution/llm_router_pool.py
import asyncio
import time
import subprocess
import psutil
import httpx
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
        # 注意：此处不更新 last_used，避免干扰空闲判断


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

        # 启动后台清理任务（定期停止空闲超时的容器）
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

    async def _start_container(self, container_name: str, port: int):
        logger.info(f"Starting container {container_name}")
        subprocess.run(["docker", "start", container_name], capture_output=True, check=False)

        # 轮询 /v1/models 端点，最多等待 60 秒
        health_url = f"http://localhost:{port}/v1/models"
        async with httpx.AsyncClient() as client:
            for _ in range(60):
                try:
                    resp = await client.get(health_url, timeout=2.0)
                    if resp.status_code == 200:
                        logger.info(f"Container {container_name} is ready")
                        return
                except Exception:
                    pass
                await asyncio.sleep(1)
        logger.warning(f"Container {container_name} did not become ready within timeout")

    async def _stop_container(self, container_name: str):
        logger.info(f"Stopping idle container {container_name}")
        subprocess.run(["docker", "stop", container_name], capture_output=True, check=False)

    async def stop_idle_containers(self, idle_seconds: int = 60):
        """主动停止所有空闲且无活动任务的容器（用于内存熔断时调用）"""
        now = time.time()
        for slot in self.model_slots.values():
            if slot.container_name and await self._is_container_running(slot.container_name):
                if slot.active_tasks == 0 and (now - slot.last_used) > idle_seconds:
                    await self._stop_container(slot.container_name)

    async def _evict_idle_containers(self, model_name: str, required_available: int) -> bool:
        """尝试驱逐空闲容器，返回是否释放了足够内存"""
        EVICTION_IDLE_THRESHOLD = 60
        now = time.time()
        candidates = []
        for name, s in self.model_slots.items():
            if name == model_name:
                continue
            if s.container_name and await self._is_container_running(s.container_name):
                if s.active_tasks == 0 and (now - s.last_used) > EVICTION_IDLE_THRESHOLD:
                    candidates.append((s.last_used, name, s))
        candidates.sort(key=lambda x: x[0])
        for _, name, s in candidates:
            logger.info(f"Evicting idle container {s.container_name} for model {name} to free memory")
            await self._stop_container(s.container_name)
            await asyncio.sleep(3)  # 等待内存释放
            mem = psutil.virtual_memory()
            if mem.available >= required_available:
                return True
        return False

    async def _ensure_memory_for_model(self, model_name: str, max_wait_seconds: int = 300) -> bool:
        """确保有足够内存启动模型，否则尝试驱逐并等待内存释放"""
        slot = self.model_slots.get(model_name)
        if not slot:
            return True

        required_gb = slot.memory_gb
        required_available = (required_gb + MEMORY_SAFETY_MARGIN_GB) * 1024 * 1024 * 1024

        start_time = time.time()
        while True:
            mem = psutil.virtual_memory()
            if mem.available >= required_available:
                return True

            if time.time() - start_time > max_wait_seconds:
                logger.error(f"Memory insufficient after waiting {max_wait_seconds}s. Required: {required_gb+MEMORY_SAFETY_MARGIN_GB}GB, available: {mem.available//(1024**3)}GB")
                return False

            # 尝试驱逐空闲容器
            evicted = await self._evict_idle_containers(model_name, required_available)
            if evicted:
                continue

            # 没有可驱逐的容器，等待其他任务释放内存
            logger.info(f"Waiting for memory to be freed... (required {required_gb+MEMORY_SAFETY_MARGIN_GB}GB, available {mem.available//(1024**3)}GB)")
            await asyncio.sleep(2)

    async def _ensure_model_ready(self, model_name: str):
        """确保模型容器正在运行，如果未运行则先检查内存并启动"""
        slot = self.model_slots.get(model_name)
        if not slot or not slot.container_name:
            return

        if not await self._is_container_running(slot.container_name):
            # 启动前检查内存
            if not await self._ensure_memory_for_model(model_name):
                raise RuntimeError(f"Not enough memory to start model {model_name}")
            await self._start_container(slot.container_name, slot.host_port)

    async def _cleanup_loop(self):
        """定期停止空闲超时的容器"""
        while True:
            await asyncio.sleep(60)
            now = time.time()
            for slot in self.model_slots.values():
                if slot.container_name and await self._is_container_running(slot.container_name):
                    if slot.active_tasks == 0 and (now - slot.last_used) > self.idle_timeout:
                        await self._stop_container(slot.container_name)

    def get_base_url(self, model_name: str) -> str:
        slot = self.model_slots.get(model_name)
        if slot and slot.host_port:
            return f"http://localhost:{slot.host_port}"
        return "http://localhost:8081"

    def get_model_load(self, model_name: str) -> float:
        """返回单个模型的负载（0-1）"""
        slot = self.model_slots.get(model_name)
        if slot and slot.max_concurrent > 0:
            return slot.active_tasks / slot.max_concurrent
        return 1.0

    def select_best_model(self, candidates: List[str]) -> Optional[str]:
        """从候选列表中选择负载最低且未满载的模型，返回模型名；如果全部满载则返回 None"""
        best = None
        best_load = 1.0
        for name in candidates:
            slot = self.model_slots.get(name)
            if slot is None:
                continue
            if slot.active_tasks < slot.max_concurrent:
                load = slot.active_tasks / slot.max_concurrent
                if load < best_load:
                    best_load = load
                    best = name
        return best

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