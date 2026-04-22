"""Execute endpoint - main entry point for AI Factory requests."""

import uuid
import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.orchestrator.graph import compile_workflow
from src.common.models import AgentResponse
from src.common.logging import setup_logging

logger = setup_logging("api.execute")

execute_router = APIRouter()

# Compiled workflow (lazy loaded)
_workflow = None


def get_workflow():
    global _workflow
    if _workflow is None:
        _workflow = compile_workflow()
    return _workflow


class ExecuteRequest(BaseModel):
    user_input: str
    session_id: Optional[str] = None
    max_retries: Optional[int] = None


@execute_router.post("")
async def execute(request: ExecuteRequest) -> AgentResponse:
    """
    Execute a user request through the AI Factory pipeline.

    The workflow: analyze -> research/code -> validate -> (retry if needed) -> final answer
    """
    if not request.user_input.strip():
        raise HTTPException(status_code=400, detail="user_input cannot be empty")

    session_id = request.session_id or uuid.uuid4().hex[:8]
    max_retries = request.max_retries or 3

    logger.info(f"Executing request for session={session_id}: {request.user_input[:150]}")

    try:
        from src.orchestrator.state import AgentState

        initial_state = AgentState(
            user_input=request.user_input,
            max_retries=max_retries,
            metadata={"session_id": session_id},
        )

        workflow = get_workflow()
        result = await workflow.ainvoke(initial_state.model_dump())

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
            code_executed=bool(execution_result),
            execution_result=execution_result,
            sources=sources,
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@execute_router.post("/stream")
async def execute_stream(request: ExecuteRequest) -> dict:
    """
    Stream execution results as they become available.
    Returns a simplified streaming response.
    """
    if not request.user_input.strip():
        raise HTTPException(status_code=400, detail="user_input cannot be empty")

    session_id = request.session_id or uuid.uuid4().hex[:8]

    logger.info(f"Streaming execution for session={session_id}")

    try:
        from src.orchestrator.state import AgentState

        initial_state = AgentState(
            user_input=request.user_input,
            max_retries=3,
            metadata={"session_id": session_id},
        )

        workflow = get_workflow()
        result = await workflow.ainvoke(initial_state.model_dump())

        return {
            "session_id": session_id,
            "success": not result.get("error"),
            "final_answer": result.get("final_answer", ""),
            "research_used": bool(result.get("research_results")),
            "code_executed": bool(result.get("execution_result")),
            "retry_count": result.get("retry_count", 0),
            "error": result.get("error"),
        }

    except Exception as e:
        logger.error(f"Streaming execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
