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

MODEL_CONFIG = {
    "Qwen3-Coder-30B-A3B-Instruct-Q5_K_M": {
        "container": "llamacpp-coder",
        "port": 8081,
        "max_concurrent": 1,
        "memory_gb": 35,
    },
    "Qwen3.6-27B-Q5_K_M": {
        "container": "llamacpp-writing",
        "port": 8082,
        "max_concurrent": 1,
        "memory_gb": 30,
    },
    "Qwen3.6-35B-A3B-UD-Q5_K_M": {
        "container": "llamacpp-default",
        "port": 8083,
        "max_concurrent": 1,
        "memory_gb": 40,
    },
    "Qwen3.5-9B-Q4_K_M": {
        "container": "llamacpp-fast",
        "port": 8084,
        "max_concurrent": 1,
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
}

MEMORY_SAFETY_MARGIN_GB = 2
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
    def __init__(self, default_timeout: int = 600, idle_timeout: int = 86400):
        self.default_timeout = default_timeout
        self.idle_timeout = idle_timeout
        self.model_slots: Dict[str, ModelSlot] = {}
        self._lock = asyncio.Lock()
        self._eviction_lock = asyncio.Lock()
        self._startup_lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self.large_model_semaphore = asyncio.Semaphore(1)

        for name, cfg in MODEL_CONFIG.items():
            self.register_model(
                name,
                max_concurrent=cfg["max_concurrent"],
                container_name=cfg.get("container"),
                host_port=cfg["port"],
                memory_gb=cfg.get("memory_gb", 30),
            )

        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    def register_model(self, name: str, max_concurrent: int = 2,
                       container_name: Optional[str] = None,
                       host_port: int = 8081, memory_gb: int = 30):
        if name not in self.model_slots:
            cfg = MODEL_CONFIG.get(name, {})
            slot = ModelSlot(
                name=name,
                max_concurrent=cfg.get("max_concurrent", max_concurrent),
                container_name=cfg.get("container", container_name),
                host_port=cfg.get("port", host_port),
                memory_gb=cfg.get("memory_gb", memory_gb),
            )
            self.model_slots[name] = slot
            logger.debug(f"Registered model {name}")

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

        health_url = f"http://localhost:{port}/v1/models"
        async with httpx.AsyncClient() as client:
            # 1. 等待 HTTP 服务可用（最多 120 秒）
            for _ in range(120):
                try:
                    resp = await client.get(health_url, timeout=2.0)
                    if resp.status_code == 200:
                        logger.info(f"Container {container_name} HTTP server ready")
                        break
                except Exception:
                    pass
                await asyncio.sleep(1)
            else:
                raise RuntimeError(f"Container {container_name} HTTP not ready after 120s")

            # 2. 等待模型真正能够推理（最多 120 次尝试，每次间隔 2 秒）
            inference_url = f"http://localhost:{port}/v1/chat/completions"
            test_payload = {
                "model": "test",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "temperature": 0
            }
            for attempt in range(60):
                try:
                    resp = await client.post(inference_url, json=test_payload, timeout=5.0)
                    if resp.status_code == 200:
                        logger.info(f"Container {container_name} model is ready for inference")
                        await asyncio.sleep(5)    # 额外缓冲，避免 503
                        return
                    elif resp.status_code == 503:
                        logger.info(f"Container {container_name} still loading (503), waiting...")
                    else:
                        logger.warning(f"Unexpected status {resp.status_code}")
                except Exception as e:
                    logger.debug(f"Inference test failed: {e}")
                await asyncio.sleep(2)
            raise RuntimeError(f"Container {container_name} did not become ready for inference within 240s")

    # ========== 新增：带重试的容器启动方法 ==========
    async def _start_container_with_retry(self, container_name: str, port: int, max_retries: int = 5):
        """启动容器，遇失败可重试（如内存不足、超时等）"""
        for attempt in range(max_retries):
            try:
                await self._start_container(container_name, port)
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Start container {container_name} failed after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Start container {container_name} failed (attempt {attempt+1}/{max_retries}): {e}, retrying in 10s...")
                await asyncio.sleep(10)

    async def _stop_container(self, container_name: str):
        logger.info(f"Stopping idle container {container_name}")
        subprocess.run(["docker", "stop", container_name], capture_output=True, check=False)
            
    async def _ensure_memory_for_model(self, model_name: str, max_wait_seconds: int = 600) -> bool:
        slot = self.model_slots.get(model_name)
        if not slot:
            return True
        required_gb = slot.memory_gb + MEMORY_SAFETY_MARGIN_GB   # MEMORY_SAFETY_MARGIN_GB 现在为 2
        required_available = required_gb * 1024 * 1024 * 1024
        start_time = time.time()
        
        while True:
            mem = psutil.virtual_memory()
            if mem.available >= required_available:
                return True
            
            if time.time() - start_time > max_wait_seconds:
                logger.error(f"Memory insufficient after {max_wait_seconds}s: need {required_gb}GB, avail {mem.available // (1024**3)}GB")
                return False
            
            # 主动驱逐所有空闲容器（不只是针对特定模型）
            evicted = await self._evict_all_idle_containers(required_available)
            if evicted:
                mem = psutil.virtual_memory()
                if mem.available >= required_available:
                    return True
                await asyncio.sleep(10)
                continue
            
            # 驱逐后仍不足：强行停止所有空闲容器（不限时长的 idle）
            for name, s in self.model_slots.items():
                if s.container_name and s.active_tasks == 0:
                    if await self._is_container_running(s.container_name):
                        logger.info(f"Force stopping idle container {s.container_name} to free memory")
                        await self._stop_container(s.container_name)
                        await asyncio.sleep(5)
            
            # 再次检查
            mem = psutil.virtual_memory()
            if mem.available >= required_available:
                return True
            
            logger.info(f"Waiting for memory... need {required_gb}GB, avail {mem.available//(1024**3)}GB")
            await asyncio.sleep(5)

    async def _evict_idle_containers(self, model_name: str, required_available: int) -> bool:
        async with self._eviction_lock:
            now = time.time()
            candidates = []
            for name, s in self.model_slots.items():
                if name == model_name or not s.container_name:
                    continue
                if await self._is_container_running(s.container_name):
                    if s.active_tasks == 0 and (now - s.last_used) > 60:
                        candidates.append((s.last_used, name, s))
            candidates.sort(key=lambda x: x[0])
            for _, name, s in candidates:
                logger.info(f"Evicting idle container {s.container_name} to free memory")
                await self._stop_container(s.container_name)
                await asyncio.sleep(5)
                mem = psutil.virtual_memory()
                if mem.available >= required_available:
                    return True
            return False

    async def _ensure_model_ready(self, model_name: str):
        slot = self.model_slots.get(model_name)
        if not slot or not slot.container_name:
            return
        if await self._is_container_running(slot.container_name):
            return
        async with self._startup_lock:
            if await self._is_container_running(slot.container_name):
                return
            if not await self._ensure_memory_for_model(model_name):
                raise RuntimeError(f"Not enough memory to start model {model_name}")
            # 修改：使用带重试的启动方法
            await self._start_container_with_retry(slot.container_name, slot.host_port)

    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(60)
            async with self._eviction_lock:
                now = time.time()
                for slot in self.model_slots.values():
                    if slot.container_name and await self._is_container_running(slot.container_name):
                        if slot.active_tasks == 0 and (now - slot.last_used) > self.idle_timeout:
                            await self._stop_container(slot.container_name)

    def get_base_url(self, model_name: str) -> str:
        slot = self.model_slots.get(model_name)
        return f"http://localhost:{slot.host_port}" if slot else "http://localhost:8081"
        
    async def call(self, model_name: str, func: Callable, *args, timeout: Optional[int] = None, **kwargs) -> Any:
        for _ in range(2):
            if model_name not in self.model_slots:
                async with self._lock:
                    if model_name not in self.model_slots:
                        self.register_model(model_name)
            slot = self.model_slots.get(model_name)
            if slot:
                break
            await asyncio.sleep(0.1)
        else:
            raise RuntimeError(f"Model slot '{model_name}' not found")

        await self._ensure_model_ready(model_name)
        kwargs['base_url'] = self.get_base_url(model_name)

        # 重试 503 / Loading model 错误
        max_retries = 3
        last_exception = None
        for attempt in range(max_retries):
            try:
                if model_name in LARGE_MODELS:
                    async with self.large_model_semaphore:
                        await slot.acquire()
                        try:
                            return await asyncio.wait_for(func(model_name, *args, **kwargs),
                                                        timeout=timeout or self.default_timeout)
                        finally:
                            slot.release()
                else:
                    await slot.acquire()
                    try:
                        return await asyncio.wait_for(func(model_name, *args, **kwargs),
                                                    timeout=timeout or self.default_timeout)
                    finally:
                        slot.release()
            except Exception as e:
                last_exception = e
                # 仅对 503 或 "Loading model" 错误进行重试
                error_str = str(e)
                if "503" in error_str or "Loading model" in error_str:
                    if attempt < max_retries - 1:
                        logger.warning(f"Model {model_name} returned {type(e).__name__}: {error_str[:100]}, retrying in 5s... (attempt {attempt+1}/{max_retries})")
                        await asyncio.sleep(5)
                        continue
                # 其他错误或最后一次重试失败，直接抛出
                break

        logger.error(f"Model {model_name} call failed after {max_retries} attempts: {type(last_exception).__name__}: {str(last_exception)}", exc_info=True)
        raise last_exception

    async def call_with_fallback(self, model_candidates: List[str], func: Callable, *args,timeout:Optional[int]=None, **kwargs) -> Any:
        last_exception = None
        for model in model_candidates:
            try:
                return await self.call(model, func, *args,timeout=timeout, **kwargs)
            except Exception as e:
                logger.warning(f"Model {model} failed: {type(e).__name__}: {str(e)}", exc_info=True)
                last_exception = e
        raise last_exception or RuntimeError("All candidate models failed")

    async def shutdown(self):
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def cleanup_all_idle_containers_force(self, idle_seconds: int = 0):
        """强制停止所有空闲容器（忽略空闲时长）。"""
        async with self._eviction_lock:
            for name, slot in self.model_slots.items():
                if slot.container_name and await self._is_container_running(slot.container_name):
                    if slot.active_tasks == 0:
                        await self._stop_container(slot.container_name)
                        logger.info(f"Forcibly stopped idle container {slot.container_name}")

    async def warmup_models(self, model_names: List[str], timeout: float = 120.0, max_memory_percent: int = 85):
        """预热指定的模型列表，顺序启动确保内存不足时不会全部失败。"""
        for model_name in model_names:
            try:
                # 检查内存是否允许
                mem = psutil.virtual_memory()
                if mem.percent > max_memory_percent:
                    logger.warning(f"Memory {mem.percent}% > {max_memory_percent}%, skipping warmup for {model_name}")
                    continue
                await self._ensure_model_ready(model_name)
                # 可选：发送一个最小 token 的请求来真正加载模型
                async def _warmup_request(model, *args, **kwargs):
                    from openai import AsyncOpenAI
                    base_url = self.get_base_url(model)
                    client = AsyncOpenAI(api_key="not-needed", base_url=base_url)
                    await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": "ping"}],
                        max_tokens=1,
                        temperature=0,
                    )
                await self.call(model_name, _warmup_request, timeout=timeout)
                logger.info(f"Warmed up model {model_name}")
            except Exception as e:
                logger.error(f"Failed to warm up model {model_name}: {e}")
            
    async def _evict_all_idle_containers(self, required_available: int) -> bool:
        """尝试驱逐所有空闲容器，返回是否释放了足够内存"""
        evicted = False
        for name, s in self.model_slots.items():
            if s.container_name and s.active_tasks == 0:
                if await self._is_container_running(s.container_name):
                    logger.info(f"Evicting idle container {s.container_name} to free memory")
                    await self._stop_container(s.container_name)
                    evicted = True
                    await asyncio.sleep(5)
                    mem = psutil.virtual_memory()
                    if mem.available >= required_available:
                        return True
        return evicted

_pool: Optional[LLMRouterPool] = None

def get_llm_router_pool() -> LLMRouterPool:
    global _pool
    if _pool is None:
        _pool = LLMRouterPool()
    return _pool