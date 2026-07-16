"""
API 路由注册中心
"""
from fastapi import APIRouter

from src.api.endpoints import execute_router, health_router

# 尝试导入验证端点（Phase 6）
try:
    from src.api.endpoints.validate import router as validate_router
    VALIDATE_AVAILABLE = True
except ImportError:
    VALIDATE_AVAILABLE = False
    validate_router = None

# 创建主路由器
router = APIRouter()

# ========== 注册子路由器（所有 include_router 必须指定非空 prefix） ==========
router.include_router(execute_router, prefix="/execute", tags=["execute"])
router.include_router(health_router, prefix="/health", tags=["health"])

if VALIDATE_AVAILABLE and validate_router is not None:
    router.include_router(validate_router, prefix="/validate", tags=["validation"])