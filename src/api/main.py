# src/api/main.py (修改后)
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import psutil

from src.config import config
from src.api.routes import router
from src.common.logging import setup_logging
from src.execution.llm_router_pool import get_llm_router_pool
from src.db import init_db_pool, close_db_pool
from src.api.endpoints.novel import router as novel_router

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
    
    # 初始化数据库连接池
    await init_db_pool()
    
    try:
        mem = psutil.virtual_memory()
        logger.info(f"Startup - Memory: {mem.percent:.1f}% used, available: {mem.available // (1024**3)}GB")
        
        pool = get_llm_router_pool()
        
        # 定义需要预热的模型列表
        warm_models = [ 
            "Qwen3-32B-Q5_K_M", 
            "Qwen3-32B-Q5_K_M-writer",
        ]
        
        # 只清理不在预热列表中的空闲容器
        cleaned_count = 0
        for model_name, slot in pool.model_slots.items():
            if model_name not in warm_models and slot.container_name:
                if await pool._is_container_running(slot.container_name) and slot.active_tasks == 0:
                    await pool._stop_container(slot.container_name)
                    cleaned_count += 1
        if cleaned_count > 0:
            logger.info(f"Cleaned {cleaned_count} idle containers not in warmup list")
        
        # 预热需要的模型（仅启动尚未运行的）
        logger.info(f"Warming up models: {warm_models}")
        await pool.warmup_models(warm_models, timeout=120.0, max_memory_percent=85)
        logger.info("Model warmup completed")
        
    except Exception as e:
        logger.error(f"Startup cleanup/warmup failed: {e}")
    
    yield
    
    # 关闭时执行
    logger.info("AI Factory shutting down...")
    await close_db_pool()

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
app.include_router(novel_router, prefix="/api/v1")   # 新增

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