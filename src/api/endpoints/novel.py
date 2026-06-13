# src/api/endpoints/novel.py
import json
import uuid
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime

from src.common.logging import setup_logging
from src.orchestrator.state import AgentState
from src.orchestrator.graph import compile_workflow
from src.db import get_db_pool
from src.writing.event_store import NarrativeEventStore
from src.writing.snapshot import SnapshotManager
from src.writing.world_state import WorldState
from src.writing.delta import StateDelta
from src.db.pool import load_writing_progress, init_writing_progress
from src.writing.causality.initializer import ensure_core_predicates
from src.orchestrator.nodes import _load_scene_plans_from_db
from src.config import config
from src.agents.planner import PlannerAgent

logger = setup_logging("api.novel")
router = APIRouter()


class ResumeRequest(BaseModel):
    novel_id: str
    from_event_id: Optional[str] = None
    fork: bool = False
    regenerate_last_scene: bool = False


class EditEventRequest(BaseModel):
    payload: dict


# ========== 辅助函数 ==========
async def ensure_task_table():
    """确保任务状态表存在"""
    pool = get_db_pool()
    if not pool:
        return
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS resume_tasks (
                task_id VARCHAR(32) PRIMARY KEY,
                novel_id VARCHAR(32) NOT NULL,
                status VARCHAR(20) NOT NULL,
                started_at TIMESTAMPTZ,
                completed_at TIMESTAMPTZ,
                error TEXT,
                progress INT DEFAULT 0,
                result JSONB
            )
        """)

async def run_resume_workflow(task_id: str, novel_id: str, initial_state: AgentState):
    """后台运行续写工作流，并更新任务状态"""
    pool = get_db_pool()
    try:
        logger.info(f"Resume workflow starting for {novel_id}, task {task_id}, "
                    f"scene_plan_list length={len(initial_state.scene_plan_list)}, "
                    f"current_scene_index={initial_state.current_scene_index}")
        
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE resume_tasks SET status = 'running', started_at = NOW() WHERE task_id = $1",
                task_id
            )
        
        workflow = compile_workflow()
        result = await workflow.ainvoke(initial_state.model_dump(), config={"recursion_limit": config.langgraph_recursion_limit})
        
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE resume_tasks SET status = 'success', completed_at = NOW(), result = $1 WHERE task_id = $2",
                json.dumps({"final_answer": result.get("final_answer", "")}), task_id
            )
        logger.info(f"Resume workflow completed for {novel_id}, task {task_id}")
    except Exception as e:
        logger.error(f"Resume workflow failed for {novel_id}, task {task_id}: {e}", exc_info=True)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE resume_tasks SET status = 'failed', completed_at = NOW(), error = $1 WHERE task_id = $2",
                str(e), task_id
            )

# ========== 路由定义 ==========
@router.post("/resume")
async def resume_novel(request: ResumeRequest, background_tasks: BackgroundTasks):
    """断点续写（新架构），返回 task_id 供查询进度"""
    pool = get_db_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database pool not initialized")
    
    await ensure_task_table()
    
    # ===== 初始化默认值 =====
    current_volume = 1
    current_chapter = 1
    current_scene_index = 0
    chapter_completed = False
    outline = None
    # ========================
    
    event_store = NarrativeEventStore(pool)
    snap_mgr = SnapshotManager(pool)

    # 1. 加载最新快照（注意：现在返回三个值）
    world_state, compressed_state, last_event_id = await snap_mgr.load_latest_snapshot(request.novel_id)
    if world_state is None:
        world_state = WorldState()
        last_event_id = 0
        compressed_state = None
        logger.info(f"No snapshot found for {request.novel_id}, starting from empty world state")
    else:
        logger.info(f"Loaded latest snapshot for {request.novel_id}, last_event_id={last_event_id}")
        if compressed_state:
            logger.info(f"Loaded compressed_state: character_intents={bool(compressed_state.character_intents)}, voice_fingerprint={bool(compressed_state.voice_fingerprint)}")
    
    # 将 compressed_state 转换为字典（如果存在）
    compressed_state_dict = None
    if compressed_state:
        if hasattr(compressed_state, 'model_dump'):
            compressed_state_dict = compressed_state.model_dump()
        elif hasattr(compressed_state, 'dict'):
            compressed_state_dict = compressed_state.dict()
        else:
            compressed_state_dict = compressed_state  # 防御性

    # 2. 加载快照之后的所有事件
    events_with_id = await event_store.get_events_since(request.novel_id, since_event_id=last_event_id)
    logger.info(f"Loaded {len(events_with_id)} events since event_id {last_event_id}")

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
            await event_store.truncate_events_after(request.novel_id, target_db_id)
            events_with_id = events_with_id[:target_index+1]
            logger.info(f"Truncated events after event_id {target_db_id} (fork=False)")
        else:
            raise HTTPException(status_code=501, detail="Fork mode not implemented yet")

    # 4. 重放事件
    for evt_id, evt in events_with_id:
        delta = StateDelta(events=[evt])
        world_state = delta.apply_to(world_state)
        last_event_id = evt_id
    logger.info(f"Replayed {len(events_with_id)} events, final last_event_id={last_event_id}")

    # ========== 确保核心谓词已投影 ==========
    await ensure_core_predicates(request.novel_id, world_state)
    # ====================================

    # 5. 从 novels 表读取大纲
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT outline FROM novels WHERE novel_id = $1",
            request.novel_id
        )
    if not row:
        raise HTTPException(status_code=404, detail="Novel metadata not found")
    outline = json.loads(row["outline"]) if row["outline"] else None
    logger.info(f"Loaded outline for novel {request.novel_id}")

    # 6. 加载进度（优先 writing_progress）
    progress = await load_writing_progress(request.novel_id)
    if progress:
        current_volume = progress["current_volume"]
        current_chapter = progress["current_chapter"]
        current_scene_index = progress["current_scene"]
        chapter_completed = progress.get("chapter_completed", False)
        logger.info(f"✅ Loaded progress from writing_progress: vol={current_volume}, ch={current_chapter}, scene={current_scene_index}, chapter_completed={chapter_completed}")
    else:
        # 回退到 novels 表
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT MAX(chapter_num) as last_chapter FROM narrative_events WHERE novel_id = $1 AND volume_num = $2",
                request.novel_id, current_volume
            )
            if row and row["last_chapter"]:
                actual_chapter = row["last_chapter"]
                if actual_chapter > current_chapter:
                    current_chapter = actual_chapter
                    current_scene_index = 0
        
        logger.info(f"⚠️ No writing_progress record, falling back to novels table: vol={current_volume}, ch={current_chapter}, scene={current_scene_index}")
        await init_writing_progress(request.novel_id, current_volume, current_chapter, current_scene_index, False)

    # 校验 current_volume 是否超出大纲卷数
    if outline:
        total_volumes = len(outline.get("volumes", []))
        if total_volumes > 0 and current_volume > total_volumes:
            logger.warning(f"current_volume {current_volume} exceeds outline volume count {total_volumes}, resetting to {total_volumes}")
            current_volume = total_volumes
            current_chapter = 1
            current_scene_index = 0
            chapter_completed = False
            await init_writing_progress(request.novel_id, current_volume, current_chapter, current_scene_index, False)
            async with pool.acquire() as conn:
                await conn.execute("""
                    UPDATE novels 
                    SET current_volume = $1, current_chapter = $2, current_scene_index = $3
                    WHERE novel_id = $4
                """, current_volume, current_chapter, current_scene_index, request.novel_id)
            logger.info(f"Reset progress to volume {current_volume}, chapter {current_chapter}")

    # 7. 从 scene_execution_units 加载当前章节的场景计划
    scene_plan_list, total_scenes = await _load_scene_plans_from_db(
        pool, request.novel_id, current_volume, current_chapter
    )
    logger.info(f"Loaded {len(scene_plan_list)} scenes from scene_execution_units")

    # 9. 计算当前卷的总章节数
    total_chapters_in_volume = 0
    if outline and "volumes" in outline:
        volumes = outline["volumes"]
        vol_idx = current_volume - 1
        if 0 <= vol_idx < len(volumes):
            total_chapters_in_volume = len(volumes[vol_idx].get("chapters", []))
            logger.info(f"Current volume {current_volume} has {total_chapters_in_volume} chapters")
        else:
            logger.warning(f"Invalid volume index {vol_idx} (total volumes={len(volumes)})")
    else:
        logger.warning("No outline or volumes found, total_chapters_in_volume will be determined later")
            
    # 10. 构造 AgentState
    initial_state = AgentState(
        user_input="继续写作",
        novel_id=request.novel_id,
        task_type="scene_plan",
        resume=True,
        outline=outline,
        current_volume=current_volume,
        current_chapter=current_chapter,
        current_scene_index=current_scene_index,
        total_scenes_in_chapter=total_scenes,
        current_state=world_state.to_dict(),
        last_sequence_id=last_event_id,
        total_chapters_in_volume=total_chapters_in_volume,
        scene_plan_list=scene_plan_list,
        compressed_state=compressed_state_dict,
        voice_memory=compressed_state_dict.get("voice_fingerprint") if compressed_state_dict else None,
    )
    logger.info(f"AgentState constructed: volume={current_volume}, chapter={current_chapter}, scene={current_scene_index}, total_scenes={total_scenes}, total_chapters_in_volume={total_chapters_in_volume}")
    
    # 11. 创建任务记录
    task_id = uuid.uuid4().hex[:12]
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO resume_tasks (task_id, novel_id, status, started_at)
            VALUES ($1, $2, 'pending', NOW())
        """, task_id, request.novel_id)
    logger.info(f"Created resume task {task_id} for novel {request.novel_id}")

    # 12. 添加后台任务
    background_tasks.add_task(run_resume_workflow, task_id, request.novel_id, initial_state)
    logger.info(f"Background task scheduled for resume workflow, task_id={task_id}")

    return {
        "task_id": task_id,
        "novel_id": request.novel_id,
        "status": "pending",
        "message": "Workflow resumed in background"
    }

@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """查询续写任务状态"""
    pool = get_db_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database pool not initialized")
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM resume_tasks WHERE task_id = $1", task_id)
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": row["task_id"],
        "novel_id": row["novel_id"],
        "status": row["status"],
        "started_at": row["started_at"].isoformat() if row["started_at"] else None,
        "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
        "error": row["error"],
        "progress": row["progress"],
        "result": row["result"]
    }


@router.get("/novel_id/{novel_id}/events")
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


@router.patch("/events/{event_uuid}")
async def edit_event(event_uuid: str, edit: EditEventRequest):
    """修改事件 payload 并删除该事件之后的所有事件及对应快照"""
    pool = get_db_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database pool not initialized")

    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                "SELECT novel_id, id FROM narrative_events WHERE event_uuid = $1",
                event_uuid
            )
            if not row:
                raise HTTPException(status_code=404, detail="Event not found")
            novel_id = row["novel_id"]
            event_db_id = row["id"]

            await conn.execute(
                "UPDATE narrative_events SET event_data = $1 WHERE event_uuid = $2",
                json.dumps(edit.payload), event_uuid
            )
            await conn.execute(
                "DELETE FROM narrative_events WHERE novel_id = $1 AND id > $2",
                novel_id, event_db_id
            )
            await conn.execute(
                "UPDATE novels SET current_state = NULL, last_sequence_id = 0 WHERE novel_id = $1",
                novel_id
            )
            await conn.execute(
                "DELETE FROM world_snapshots WHERE novel_id = $1",
                novel_id
            )

    return {"status": "event_updated", "novel_id": novel_id}

@router.get("/novel_id/{novel_id}/progress")
async def get_novel_progress(novel_id: str):
    """获取小说当前写作进度（卷、章、场景）"""
    pool = get_db_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database pool not initialized")
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT current_volume, current_chapter, current_scene, chapter_completed FROM writing_progress WHERE project_id = $1",
            novel_id
        )
        if row:
            return {
                "current_volume": row["current_volume"],
                "current_chapter": row["current_chapter"],
                "current_scene": row["current_scene"],
                "chapter_completed": row["chapter_completed"]
            }
        row = await conn.fetchrow(
            "SELECT current_volume, current_chapter, current_scene_index FROM novels WHERE novel_id = $1",
            novel_id
        )
        if not row:
            raise HTTPException(status_code=404, detail="Novel not found")
        return {
            "current_volume": row["current_volume"],
            "current_chapter": row["current_chapter"],
            "current_scene": row["current_scene_index"],
            "chapter_completed": False
        }
        
@router.get("/novel_id/{novel_id}/outline")
async def get_novel_outline(novel_id: str):
    """获取小说大纲（检查是否存在）"""
    pool = get_db_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database pool not initialized")
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT outline FROM novels WHERE novel_id = $1",
            novel_id
        )
    if not row:
        raise HTTPException(status_code=404, detail="Novel not found")
    return {"outline": row["outline"] is not None, "has_outline": row["outline"] is not None}


class SingleVolumeRequest(BaseModel):
    novel_id: str
    volume_num: int
    total_chapters: int = 100
    volume_title: Optional[str] = None
    target_realm: Optional[str] = None
    core_conflict: Optional[str] = None


@router.post("/novel/volume/chapters")
async def generate_volume_chapters(request: SingleVolumeRequest):
    pool = get_db_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database pool not initialized")

    async with pool.acquire() as conn:
        async with conn.transaction():
            # 1. 加载最新大纲（加行锁，防止并发修改，但实际串行调用不需要）
            row = await conn.fetchrow(
                "SELECT outline, revision FROM novels WHERE novel_id = $1 FOR UPDATE",
                request.novel_id
            )
            if not row:
                outline = {
                    "title": "修仙长路",
                    "world_rules": [],
                    "characters": [{"name": "林逸", "initial_state": {"realm": "炼气", "level": 1}}],
                    "volumes": []
                }
                revision = 0
            else:
                outline = json.loads(row["outline"]) if row["outline"] else {
                    "title": "修仙长路",
                    "world_rules": [],
                    "characters": [{"name": "林逸", "initial_state": {"realm": "炼气", "level": 1}}],
                    "volumes": []
                }
                revision = row["revision"]

            # 2. 检查该卷是否已存在且完整
            existing_volumes = outline.get("volumes", [])
            for vol in existing_volumes:
                if vol.get("volume_num") == request.volume_num:
                    chapters = vol.get("chapters", [])
                    if len(chapters) >= request.total_chapters:
                        return {"status": "already_complete", "volume_num": request.volume_num}

            # 3. 生成章节列表
            temp_state = AgentState(
                novel_id=request.novel_id,
                task_type="chapters_outline",
                user_input="",
                project_id="temp",
                metadata={
                    "current_volume_info": {
                        "volume_num": request.volume_num,
                        "title": request.volume_title or f"第{request.volume_num}卷",
                        "target_realm": request.target_realm or "筑基",
                        "core_conflict": request.core_conflict or "默认冲突"
                    },
                    "chapters_per_vol": request.total_chapters,
                    "chapter_range": (1, request.total_chapters)
                }
            )
            chapters = await _generate_chapters_for_volume(
                volume_num=request.volume_num,
                volume_title=request.volume_title,
                target_realm=request.target_realm,
                core_conflict=request.core_conflict,
                total_chapters=request.total_chapters,
                state=temp_state
            )

            # 4. 合并到 outline
            new_volume = {
                "volume_num": request.volume_num,
                "title": request.volume_title or f"第{request.volume_num}卷",
                "target_realm": request.target_realm or "筑基",
                "core_conflict": request.core_conflict or "默认冲突",
                "chapters": chapters
            }
            updated = False
            for i, vol in enumerate(existing_volumes):
                if vol.get("volume_num") == request.volume_num:
                    existing_volumes[i] = new_volume
                    updated = True
                    break
            if not updated:
                existing_volumes.append(new_volume)
            outline["volumes"] = existing_volumes

            # 5. 保存到数据库（直接覆盖，不检查乐观锁）
            await conn.execute("""
                INSERT INTO novels (novel_id, outline, revision, updated_at)
                VALUES ($1, $2, 1, NOW())
                ON CONFLICT (novel_id) DO UPDATE
                SET outline = EXCLUDED.outline, revision = novels.revision + 1, updated_at = NOW()
            """, request.novel_id, json.dumps(outline))

            return {"status": "success", "volume_num": request.volume_num, "chapters_count": len(chapters)}


@router.get("/novel_id/{novel_id}/outline/detail")
async def get_novel_outline_detail(novel_id: str):
    """获取小说完整大纲"""
    pool = get_db_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database pool not initialized")
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT outline FROM novels WHERE novel_id = $1", novel_id)
    if not row or not row["outline"]:
        raise HTTPException(status_code=404, detail="Outline not found")
    return {"outline": json.loads(row["outline"])}


async def _generate_chapters_for_volume(volume_num: int, volume_title: str, target_realm: str,
                                         core_conflict: str, total_chapters: int, state: AgentState) -> list:
    from src.agents.planner import PlannerAgent
    from src.prompts.planner_prompts import PROMPT_REGISTRY

    planner = PlannerAgent()
    state.metadata["current_volume_info"] = {
        "volume_num": volume_num,
        "title": volume_title,
        "target_realm": target_realm,
        "core_conflict": core_conflict
    }
    state.metadata["chapters_per_vol"] = total_chapters
    state.metadata["chapter_range"] = (1, total_chapters)

    builder = PROMPT_REGISTRY.get("chapters_outline")
    prompt = builder.build(state)
    response = await planner.plan_request_with_prompt(prompt, "chapters_outline")
    result = builder.parse_response(response)
    chapters = result.get("chapters", [])

    # 补全缺失的章节
    while len(chapters) < total_chapters:
        missing_idx = len(chapters) + 1
        chapters.append({
            "chapter_num": missing_idx,
            "title": f"第{missing_idx}章",
            "must_events": [],
            "forbidden_events": []
        })
    # 修正序号
    for i, ch in enumerate(chapters):
        ch["chapter_num"] = i + 1

    return chapters