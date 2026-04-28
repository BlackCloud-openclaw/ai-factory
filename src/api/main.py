# src/api/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.api.routes import router
from src.common.logging import setup_logging

logger = setup_logging("api.main")

# ========== 请求体大小限制中间件 ==========
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_size: int = 1_000_000):  # 默认 1MB
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
    # 可选：启动 MemoryGuard, Watchdog 等安全组件
    # from src.security.memory_guard import get_memory_guard
    # await get_memory_guard().start()
    yield
    # 关闭时执行
    logger.info("AI Factory shutting down...")
    # 可选：关闭 LLMRouterPool 等
    # from src.execution.llm_router_pool import get_llm_router_pool
    # await get_llm_router_pool().shutdown()

# ========== 创建 FastAPI 应用 ==========
app = FastAPI(
    title="AI Factory",
    description="AI-powered agent system with LangGraph orchestration, RAG knowledge base, and code execution sandbox.",
    version="1.0.0",
    lifespan=lifespan,
)

# 添加 CORS 中间件（允许跨域，可根据需要调整）
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

# 添加健康检查端点（也可以在 routes/health.py 中定义，这里直接添加作为示例）
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ai-factory"}

@app.get("/ready")
async def readiness():
    return {"status": "ready"}

@app.get("/live")
async def liveness():
    return {"status": "alive"}