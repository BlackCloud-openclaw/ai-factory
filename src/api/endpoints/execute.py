"""Execute endpoint - main entry point for AI Factory requests."""

import uuid
import asyncio
import time
import json
import logging
import psutil
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Request

from src.orchestrator.state import AgentState
from src.db import get_db_pool
from src.orchestrator.graph import compile_workflow
from src.common.models import AgentResponse
from src.common.logging import setup_logging
from src.execution.llm_router_pool import get_llm_router_pool
from src.api.scheduler import get_scheduler

# 新架构事件存储和快照
from src.writing.event_store import NarrativeEventStore
from src.writing.snapshot import SnapshotManager
from src.writing.world_state import WorldState
from src.writing.delta import StateDelta

# 全局并发控制（最多同时运行 1 个工作流，防止资源竞争和文件覆盖）
_global_workflow_semaphore = asyncio.Semaphore(1)

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
    task_type: str = "code"           # "code" / "novel_outline" / "scene_plan"
    novel_id: Optional[str] = None    # 用于续写时指定哪部小说
    resume: bool = False              # 是否从上次中断处继续


async def _run_workflow(req: ExecuteRequest) -> dict:
    logger = logging.getLogger("api.execute")

    session_id = req.session_id or uuid.uuid4().hex[:8]
    project_id = req.project_id or session_id
    max_retries = req.max_retries or 3

    initial_state = AgentState(
        user_input=req.user_input,
        project_id=project_id,
        max_retries=max_retries,
        metadata={"session_id": session_id, "project_id": project_id},
        task_type=req.task_type,
        novel_id=req.novel_id,
        resume=req.resume,
    )

    pool = get_db_pool()

    # ========== 断点续写模式（基于新架构） ==========
    if req.resume and req.novel_id and pool:
        try:
            event_store = NarrativeEventStore(pool)
            snap_mgr = SnapshotManager(pool)

            # 1. 加载最新快照
            world_state, _, last_event_id = await snap_mgr.load_latest_snapshot(req.novel_id)
            if world_state is None:
                world_state = WorldState()
                last_event_id = 0

            # 2. 加载快照之后的事件
            events_with_id = await event_store.get_events_since(req.novel_id, since_event_id=last_event_id)

            # 3. 重放事件，更新世界状态
            for evt_id, evt in events_with_id:
                delta = StateDelta(events=[evt])
                world_state = delta.apply_to(world_state)
                last_event_id = evt_id

            # 4. 保存当前状态到 initial_state
            initial_state.current_state = world_state.to_dict()
            initial_state.last_sequence_id = last_event_id

            # 5. 从 novels 表读取元数据（大纲、进度等）
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """SELECT outline, current_volume, current_chapter, current_scene_index
                    FROM novels WHERE novel_id = $1""",
                    req.novel_id
                )
                
                if row:
                    if row["outline"]:
                        initial_state.outline = json.loads(row["outline"])
                    initial_state.current_volume = row["current_volume"] or 1
                    initial_state.current_chapter = row["current_chapter"] or 1
                    initial_state.current_scene_index = row["current_scene_index"] or 0
                    
                    initial_state.metadata["scene_plan_list"] = []
                    initial_state.metadata["total_scenes_in_chapter"] = 0
                    initial_state.metadata["current_scene_index"] = initial_state.current_scene_index
                    initial_state.metadata["current_scene_plan"] = None
                    initial_state.scene_plan_list = []
                    initial_state.total_scenes_in_chapter = 0

                    if initial_state.outline and "volumes" in initial_state.outline:
                        volumes = initial_state.outline["volumes"]
                        vol_idx = initial_state.current_volume - 1
                        if 0 <= vol_idx < len(volumes):
                            total_chapters = len(volumes[vol_idx].get("chapters", []))
                            initial_state.total_chapters_in_volume = total_chapters

            logger.info(f"Resume: restored state for {req.novel_id}, "
                        f"volume={initial_state.current_volume}, "
                        f"chapter={initial_state.current_chapter}, "
                        f"scene={initial_state.current_scene_index}, "
                        f"last_event_id={last_event_id}")

        except Exception as e:
            logger.error(f"Failed to resume state for novel {req.novel_id}: {e}", exc_info=True)
            initial_state.resume = False

    # ========== 非续写模式：加载大纲和已有进度 ==========
    elif not req.resume and req.task_type in ("scene_plan", "novel_outline") and req.novel_id and pool:
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """SELECT outline, current_volume, current_chapter, current_scene_index
                       FROM novels WHERE novel_id = $1""",
                    req.novel_id
                )
                if row:
                    if row["outline"]:
                        initial_state.outline = json.loads(row["outline"])
                    initial_state.current_volume = row["current_volume"] or 1
                    initial_state.current_chapter = row["current_chapter"] or 1
                    initial_state.current_scene_index = row["current_scene_index"] or 0
                    
                    initial_state.metadata["scene_plan_list"] = []
                    initial_state.metadata["total_scenes_in_chapter"] = 0
                    initial_state.metadata["current_scene_index"] = initial_state.current_scene_index
                    initial_state.metadata["current_scene_plan"] = None
                    initial_state.scene_plan_list = []
                    initial_state.total_scenes_in_chapter = 0

                    if initial_state.outline and "volumes" in initial_state.outline:
                        volumes = initial_state.outline["volumes"]
                        vol_idx = initial_state.current_volume - 1
                        if 0 <= vol_idx < len(volumes):
                            total_chapters = len(volumes[vol_idx].get("chapters", []))
                            initial_state.total_chapters_in_volume = total_chapters

                    logger.info(f"Loaded outline for novel {req.novel_id} (non-resume mode)")
        except Exception as e:
            logger.error(f"Failed to load outline for novel {req.novel_id}: {e}", exc_info=True)

    workflow = get_workflow()
    result = await asyncio.wait_for(
        workflow.ainvoke(initial_state.model_dump(), config={"recursion_limit": 500}),
        timeout=3600,
    )
    return result


@execute_router.post("")
async def execute(req: ExecuteRequest, request: Request) -> AgentResponse:
    # 记录客户端 IP 和请求内容（用于排查重复请求）
    logger.info(f"Request from {request.client.host}: {req.user_input[:100]}")
    
    global _last_memory_cleanup

    if not req.user_input.strip():
        raise HTTPException(status_code=400, detail="user_input cannot be empty")

    mem = psutil.virtual_memory()
    if mem.percent > 90:
        pool = get_llm_router_pool()
        await pool.cleanup_all_idle_containers_force()
        mem = psutil.virtual_memory()
        if mem.percent > 90:
            raise HTTPException(status_code=503, detail="System memory overloaded, please retry later")

    async with _global_workflow_semaphore:
        session_id = req.session_id or uuid.uuid4().hex[:8]
        project_id = req.project_id or session_id
        logger.info(f"Executing request for session={session_id}, project={project_id}: {req.user_input[:150]}")

        mem = psutil.virtual_memory()
        pool = get_llm_router_pool()

        if mem.percent > 90:
            now = time.time()
            if now - _last_memory_cleanup > 30:
                await pool.cleanup_all_idle_containers_force()
                _last_memory_cleanup = now
                mem = psutil.virtual_memory()

        if mem.percent > 96:
            logger.warning(f"Memory overloaded: {mem.percent}%, rejecting request")
            raise HTTPException(status_code=503, detail="System memory overloaded, please retry later")

        lower_input = req.user_input.lower()
        if any(kw in lower_input for kw in ["写代码", "函数", "斐波那契", "计算"]):
            priority = 1
        elif any(kw in lower_input for kw in ["写小说", "故事", "雨夜"]):
            priority = 3
        else:
            priority = 2

        scheduler = get_scheduler()
        try:
            result = await scheduler.submit(priority, _run_workflow, req)
        except Exception as e:
            logger.error(f"Scheduler submission failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

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