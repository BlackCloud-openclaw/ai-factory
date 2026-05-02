# src/api/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import psutil

from src.api.routes import router
from src.common.logging import setup_logging
from src.execution.llm_router_pool import get_llm_router_pool

logger = setup_logging("api.main")

# ========== 请求体大小限制中间件 ==========
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_size: int = 1_000_000):
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_size:
            raise HTTPException(status_code=413, detail="Request body too large")
        return await call_next(request)

# ========== 速率限制器 ==========
limiter = Limiter(key_func=get_remote_address)

# ========== 生命周期管理 ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    logger.info("AI Factory starting up...")
    
    # 检查内存占用率，如果过高则清理空闲的 LLM 容器
    try:
        mem = psutil.virtual_memory()
        logger.info(f"Startup - Memory: {mem.percent:.1f}% used, available: {mem.available // (1024**3)}GB")
        if mem.percent > 20.0:
            logger.warning(f"High memory usage ({mem.percent:.1f}%) on startup, cleaning idle LLM containers")
            pool = get_llm_router_pool()
            await pool.cleanup_all_idle_containers_force(idle_seconds=0)
            mem2 = psutil.virtual_memory()
            logger.info(f"After cleanup - Memory: {mem2.percent:.1f}% used, available: {mem2.available // (1024**3)}GB")
        else:
            logger.info("Memory usage is acceptable, no container cleanup needed")
    except Exception as e:
        logger.error(f"Startup memory cleanup failed: {e}")
    
     # ========== 预热常用模型（新增） ==========
    try:
        pool = get_llm_router_pool()
        # 选择你最常用的 1-2 个模型，例如代码模型和写作模型
        warm_models = [
            "Qwen3.6-35B-A3B-UD-Q5_K_M", 
            #"Qwen3.6-27B-Q5_K_M",   
            "Qwen2.5-Coder-32B-Instruct-Q5_K_M",
        ]
        logger.info(f"Warming up models: {warm_models}")
        # 使用 warmup_models 方法（顺序启动，避免内存瞬间占满）
        await pool.warmup_models(warm_models, timeout=120.0, max_memory_percent=85)
        logger.info("Model warmup completed")
    except Exception as e:
        logger.error(f"Model warmup failed: {e}")
        # 预热失败不阻止应用启动，模型将按需启动（可能遇到冷启动延迟）
    
    yield
    
    # 关闭时执行
    logger.info("AI Factory shutting down...")
    # 可选：优雅关闭调度器和路由池
    # from src.api.scheduler import get_scheduler
    # await get_scheduler().shutdown()
    # await get_llm_router_pool().shutdown()

# ========== 创建 FastAPI 应用 ==========
app = FastAPI(
    title="AI Factory",
    description="AI-powered agent system with LangGraph orchestration, RAG knowledge base, and code execution sandbox.",
    version="1.0.0",
    lifespan=lifespan,
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加请求体大小限制中间件
app.add_middleware(RequestSizeLimitMiddleware, max_size=1_000_000)

# 设置速率限制异常处理
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 注册路由
app.include_router(router, prefix="/api/v1")

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ai-factory"}

@app.get("/ready")
async def readiness():
    return {"status": "ready"}

@app.get("/live")
async def liveness():
    return {"status": "alive"}