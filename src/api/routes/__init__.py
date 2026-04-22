from fastapi import APIRouter

from src.api.endpoints import execute_router, health_router

router = APIRouter()
router.include_router(execute_router, prefix="/execute", tags=["execute"])
router.include_router(health_router, tags=["health"])

__all__ = ["router"]
