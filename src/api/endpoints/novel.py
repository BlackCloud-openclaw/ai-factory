# src/api/endpoints/novel.py
import json
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from src.common.logging import setup_logging
from src.orchestrator.state import AgentState
from src.orchestrator.graph import compile_workflow
from src.db import get_db_pool
from src.writing.event_store import NarrativeEventStore
from src.writing.snapshot import SnapshotManager
from src.writing.world_state import WorldState
from src.writing.delta import StateDelta

logger = setup_logging("api.novel")
router = APIRouter()


class ResumeRequest(BaseModel):
    novel_id: str
    from_event_id: Optional[str] = None   # event_uuid
    fork: bool = False
    regenerate_last_scene: bool = False


class EditEventRequest(BaseModel):
    payload: dict


# ========== 路由定义 ==========

@router.post("/novel/resume")
async def resume_novel(request: ResumeRequest, background_tasks: BackgroundTasks):
    """断点续写（新架构）"""
    pool = get_db_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database pool not initialized")

    event_store = NarrativeEventStore(pool)
    snap_mgr = SnapshotManager(pool)

    # 1. 加载最新快照
    world_state, _, last_event_id = await snap_mgr.load_latest_snapshot(request.novel_id)
    if world_state is None:
        world_state = WorldState()
        last_event_id = 0

    # 2. 加载快照之后的所有事件
    events_with_id = await event_store.get_events_since(request.novel_id, since_event_id=last_event_id)

    # 3. 如果需要从特定事件分叉/截断
    if request.from_event_id:
        target_index = None
        target_db_id = None
        for idx, (evt_id, evt) in enumerate(events_with_id):
            if evt.event_id == request.from_event_id:
                target_index = idx
                target_db_id = evt_id
                break
        if target_index is None:
            raise HTTPException(status_code=404, detail=f"Event {request.from_event_id} not found")
        if not request.fork:
            # 截断：删除该事件之后的所有事件
            await event_store.truncate_events_after(request.novel_id, target_db_id)
            events_with_id = events_with_id[:target_index+1]
        else:
            raise HTTPException(status_code=501, detail="Fork mode not implemented yet")

    # 4. 重放事件
    for evt_id, evt in events_with_id:
        delta = StateDelta(events=[evt])
        world_state = delta.apply_to(world_state)
        last_event_id = evt_id

    # 5. 从 novels 表读取元数据
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT outline, current_volume, current_chapter, current_scene_index
               FROM novels WHERE novel_id = $1""",
            request.novel_id
        )
    if not row:
        raise HTTPException(status_code=404, detail="Novel metadata not found")

    outline = json.loads(row["outline"]) if row["outline"] else None
    current_volume = row["current_volume"] or 1
    current_chapter = row["current_chapter"] or 1
    current_scene_index = row["current_scene_index"] or 0

    # 6. 构造 AgentState
    initial_state = AgentState(
        user_input="继续写作",
        novel_id=request.novel_id,
        task_type="scene_plan",
        resume=True,
        outline=outline,
        current_volume=current_volume,
        current_chapter=current_chapter,
        current_scene_index=current_scene_index,
        current_state=world_state.to_dict(),
        last_sequence_id=last_event_id,
        metadata={"resumed": True, "from_event": request.from_event_id}
    )

    workflow = compile_workflow()

    async def run_workflow():
        try:
            await workflow.ainvoke(initial_state.model_dump(), config={"recursion_limit": 100})
            logger.info(f"Resume workflow completed for {request.novel_id}")
        except Exception as e:
            logger.error(f"Resume workflow failed: {e}")

    background_tasks.add_task(run_workflow)

    return {
        "status": "started",
        "novel_id": request.novel_id,
        "message": "Workflow resumed in background"
    }


@router.get("/novel/{novel_id}/events")
async def list_events(novel_id: str, limit: int = 100, offset: int = 0):
    """获取指定小说的事件列表（基于 narrative_events）"""
    pool = get_db_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database pool not initialized")
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, event_uuid, event_type, event_data, event_version, timestamp
               FROM narrative_events
               WHERE novel_id = $1
               ORDER BY id ASC
               LIMIT $2 OFFSET $3""",
            novel_id, limit, offset
        )
    return [
        {
            "id": row["id"],
            "event_uuid": row["event_uuid"],
            "type": row["event_type"],
            "data": row["event_data"],
            "version": row["event_version"],
            "created_at": row["timestamp"].isoformat(),
        }
        for row in rows
    ]


@router.patch("/novel/events/{event_uuid}")
async def edit_event(event_uuid: str, edit: EditEventRequest):
    """修改事件 payload 并删除该事件之后的所有事件及对应快照"""
    pool = get_db_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database pool not initialized")

    async with pool.acquire() as conn:
        async with conn.transaction():
            # 查找事件
            row = await conn.fetchrow(
                "SELECT novel_id, id FROM narrative_events WHERE event_uuid = $1",
                event_uuid
            )
            if not row:
                raise HTTPException(status_code=404, detail="Event not found")
            novel_id = row["novel_id"]
            event_db_id = row["id"]

            # 更新事件数据
            await conn.execute(
                "UPDATE narrative_events SET event_data = $1 WHERE event_uuid = $2",
                json.dumps(edit.payload), event_uuid
            )
            # 删除该事件之后的所有事件
            await conn.execute(
                "DELETE FROM narrative_events WHERE novel_id = $1 AND id > $2",
                novel_id, event_db_id
            )
            # 清空 novels 表中的快照和场景计划
            await conn.execute(
                "UPDATE novels SET current_state = NULL, last_sequence_id = 0, scene_plan_list = NULL WHERE novel_id = $1",
                novel_id
            )
            # 删除世界快照（强制下次重新生成）
            await conn.execute(
                "DELETE FROM world_snapshots WHERE novel_id = $1",
                novel_id
            )

    return {"status": "event_updated", "novel_id": novel_id}