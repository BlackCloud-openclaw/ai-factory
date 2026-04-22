"""Health check endpoint."""

from fastapi import APIRouter

health_router = APIRouter()


@health_router.get("/check")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ai-factory",
    }


@health_router.get("/ready")
async def readiness():
    """Readiness probe."""
    return {"status": "ready"}


@health_router.get("/live")
async def liveness():
    """Liveness probe."""
    return {"status": "alive"}
