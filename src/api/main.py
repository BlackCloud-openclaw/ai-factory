"""FastAPI application for AI Factory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import config
from src.common.logging import setup_logging
from src.api.routes import router

logger = setup_logging("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    logger.info("AI Factory starting up...")
    yield
    logger.info("AI Factory shutting down...")


app = FastAPI(
    title="AI Factory",
    description="AI-powered agent system with LangGraph orchestration, RAG knowledge base, and code execution sandbox.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health():
    """Root health check."""
    return {"status": "healthy", "service": "ai-factory"}


@app.get("/ready")
async def readiness():
    """Readiness probe."""
    return {"status": "ready"}
