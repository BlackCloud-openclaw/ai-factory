"""Execute endpoint - main entry point for AI Factory requests."""

import uuid
import asyncio
import time
from typing import Optional

import psutil
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.orchestrator.graph import compile_workflow
from src.common.models import AgentResponse
from src.common.logging import setup_logging
from src.execution.llm_router_pool import get_llm_router_pool
from src.api.scheduler import get_scheduler

logger = setup_logging("api.execute")

execute_router = APIRouter()

# Compiled workflow (lazy loaded)
_workflow = None

# 内存熔断冷却（上次清理时间）
_last_memory_cleanup = 0


def get_workflow():
    global _workflow
    if _workflow is None:
        _workflow = compile_workflow()
    return _workflow


class ExecuteRequest(BaseModel):
    user_input: str
    session_id: Optional[str] = None
    project_id: Optional[str] = None
    max_retries: Optional[int] = None


async def _run_workflow(request: ExecuteRequest) -> dict:
    """实际执行工作流的协程，供调度器调用"""
    from src.orchestrator.state import AgentState

    session_id = request.session_id or uuid.uuid4().hex[:8]
    project_id = request.project_id or session_id
    max_retries = request.max_retries or 3

    initial_state = AgentState(
        user_input=request.user_input,
        project_id=project_id,
        max_retries=max_retries,
        metadata={"session_id": session_id, "project_id": project_id},
    )

    workflow = get_workflow()
    # 总超时 3600 秒（1小时）
    result = await asyncio.wait_for(
        workflow.ainvoke(initial_state.model_dump(), config={"recursion_limit": 100}),
        timeout=3600,
    )
    return result


@execute_router.post("")
async def execute(request: ExecuteRequest) -> AgentResponse:
    """
    Execute a user request through the AI Factory pipeline.
    """
    if not request.user_input.strip():
        raise HTTPException(status_code=400, detail="user_input cannot be empty")

    session_id = request.session_id or uuid.uuid4().hex[:8]
    project_id = request.project_id or session_id
    logger.info(f"Executing request for session={session_id}, project={project_id}: {request.user_input[:150]}")

    # ========== 内存熔断（两级保护） ==========
    global _last_memory_cleanup
    mem = psutil.virtual_memory()
    pool = get_llm_router_pool()

    # 软熔断：内存使用率 > 88%，尝试清理空闲容器
    if mem.percent > 88:
        now = time.time()
        if now - _last_memory_cleanup > 30:   # 每 30 秒最多清理一次
            await pool.stop_idle_containers()
            _last_memory_cleanup = now
            mem = psutil.virtual_memory()     # 重新获取内存状态

    # 硬熔断：内存使用率 > 92%，直接拒绝新请求
    if mem.percent > 92:
        logger.warning(f"Memory overloaded: {mem.percent}%, rejecting request")
        raise HTTPException(status_code=503, detail="System memory overloaded, please retry later")

    # ========== 通过优先级调度器提交任务 ==========
    # 根据用户输入简单判断优先级（数字越小优先级越高）
    lower_input = request.user_input.lower()
    if any(kw in lower_input for kw in ["写代码", "函数", "斐波那契", "计算"]):
        priority = 1   # 代码生成高优先级
    elif any(kw in lower_input for kw in ["写小说", "故事", "雨夜"]):
        priority = 3   # 写作低优先级
    else:
        priority = 2

    scheduler = get_scheduler()
    try:
        result = await scheduler.submit(priority, _run_workflow, request)
    except Exception as e:
        logger.error(f"Scheduler submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # ========== 构建响应 ==========
    execution_result = result.get("execution_result")
    sources = []
    for rr in result.get("research_results", []):
        if isinstance(rr, dict):
            for src in rr.get("sources", []):
                sources.append(src)

    return AgentResponse(
        success=not result.get("error"),
        answer=result.get("final_answer", ""),
        research_used=bool(result.get("research_results")),
        code_executed=bool(result.get("code_generated")) or bool(execution_result),
        execution_result=execution_result,
        sources=sources,
        error=result.get("error"),
    )