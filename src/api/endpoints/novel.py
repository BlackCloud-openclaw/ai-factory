# src/api/endpoints/novel.py
import json
import uuid
import logging
from pathlib import Path          # ← 新增
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime

from src.common.logging import setup_logging
from src.orchestrator.state import AgentState
from src.orchestrator.graph import compile_workflow
from src.db import get_db_pool
from src.writing.event_store import NarrativeEventStore
from src.writing.snapshot_manager import SnapshotManager
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


# ============================================================
# 重跑章节 API（Phase 15.9 运维工具）
# ============================================================

class RerunChapterRequest(BaseModel):
    novel_id: str
    volume_num: int
    chapter_num: int
    dry_run: bool = False           # True 只打印将做什么，不实际改
    keep_files: bool = False        # True 保留主/shadow 文件（仅清 DB）
    max_progress_check: bool = True # True 时禁止回滚到已生成章节之后


@router.post("/novel/rerun_chapter")
async def rerun_chapter(request: RerunChapterRequest, background_tasks: BackgroundTasks):
    """
    重跑指定章节。

    步骤：
    1. 校验 novel_id 存在
    2. 校验目标章节 <= 当前进度
    3. 回滚 writing_progress 到目标章 scene 0
    4. 删除 scene_execution_units 中该章所有行
    5. 备份并删除该章主文件 / shadow 文件（keep_files=False 时）
    6. 后台触发 resume workflow
    """
    pool = get_db_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database pool not initialized")

    novel_id = request.novel_id
    volume_num = request.volume_num
    chapter_num = request.chapter_num

    # ---------- 1. 校验进度 ----------
    progress = await load_writing_progress(novel_id)
    if not progress:
        raise HTTPException(status_code=404, detail="writing_progress not found for novel")

    cur_vol = progress["current_volume"]
    cur_ch = progress["current_chapter"]

    if request.max_progress_check:
        if (volume_num, chapter_num) > (cur_vol, cur_ch):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot rerun future chapter. Current progress: vol={cur_vol}, ch={cur_ch}"
            )

    logger.info(
        f"[RerunChapter] requested: novel={novel_id}, target=v{volume_num}c{chapter_num}, "
        f"current=v{cur_vol}c{cur_ch}, dry_run={request.dry_run}, keep_files={request.keep_files}"
    )

    # ---------- 2. 查 DB 有多少条 scene_execution_units ----------
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT scene_index, status FROM scene_execution_units
            WHERE novel_id=$1 AND volume_num=$2 AND chapter_num=$3
            ORDER BY scene_index
            """,
            novel_id, volume_num, chapter_num
        )

    existing_scenes = [(r["scene_index"], r["status"]) for r in rows]
    logger.info(f"[RerunChapter] scene_execution_units has {len(existing_scenes)} rows: {existing_scenes}")

    # ---------- 3. 目标文件路径 ----------
    main_file = Path(f"data/novels/{novel_id}/vol_{volume_num:03d}/chap_{chapter_num:03d}.txt")
    shadow_file = Path(f"data/novels/{novel_id}/shadow/vol_{volume_num:03d}/chap_{chapter_num:03d}.txt")

    main_exists = main_file.exists()
    shadow_exists = shadow_file.exists()

    # ---------- 4. dry_run ----------
    if request.dry_run:
        return {
            "dry_run": True,
            "novel_id": novel_id,
            "target": {"volume_num": volume_num, "chapter_num": chapter_num},
            "current_progress": {"volume_num": cur_vol, "chapter_num": cur_ch},
            "would_delete": {
                "scene_execution_units_rows": len(existing_scenes),
                "main_file": str(main_file) if main_exists else None,
                "shadow_file": str(shadow_file) if shadow_exists else None,
            },
            "would_set_progress_to": {
                "volume_num": volume_num,
                "chapter_num": chapter_num,
                "scene": 0,
                "chapter_completed": False,
            },
            "keep_files": request.keep_files,
        }

    # ---------- 5. 备份文件 ----------
    backup_files = []
    if not request.keep_files:
        for fp in (main_file, shadow_file):
            if fp.exists():
                bak = fp.with_suffix(fp.suffix + ".bak_rerun")
                # 若 bak 已存在，加时间戳
                if bak.exists():
                    from datetime import datetime as _dt
                    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
                    bak = fp.with_suffix(fp.suffix + f".bak_rerun.{ts}")
                fp.rename(bak)
                backup_files.append(str(bak))
                logger.info(f"[RerunChapter] backed up {fp} -> {bak}")

    # ---------- 6. 事务：清 DB + 回滚进度 ----------
    async with pool.acquire() as conn:
        async with conn.transaction():
            # 6.1 删除 scene_execution_units
            del_result = await conn.execute(
                """
                DELETE FROM scene_execution_units
                WHERE novel_id=$1 AND volume_num=$2 AND chapter_num=$3
                """,
                novel_id, volume_num, chapter_num
            )

            # 6.2 更新 writing_progress
            await conn.execute(
                """
                INSERT INTO writing_progress
                    (project_id, current_volume, current_chapter, current_scene, chapter_completed, last_updated)
                VALUES ($1, $2, $3, 0, FALSE, NOW())
                ON CONFLICT (project_id) DO UPDATE SET
                    current_volume = EXCLUDED.current_volume,
                    current_chapter = EXCLUDED.current_chapter,
                    current_scene = 0,
                    chapter_completed = FALSE,
                    last_updated = NOW()
                """,
                novel_id, volume_num, chapter_num
            )

            # 6.3 更新 novels 表（可选，保持一致性）
            await conn.execute(
                """
                UPDATE novels
                SET current_volume=$1, current_chapter=$2, current_scene_index=0
                WHERE novel_id=$3
                """,
                volume_num, chapter_num, novel_id
            )

    logger.info(
        f"[RerunChapter] DB cleaned: {del_result}, "
        f"progress rolled back to v{volume_num}c{chapter_num}s0"
    )

    # ---------- 7. 后台触发 resume ----------
    initial_state = AgentState(
        user_input="继续写作",
        novel_id=novel_id,
        task_type="scene_plan",
        resume=True,
    )
    # 从 DB 加载必要状态（复用 resume_novel 的逻辑，或直接调用它）
    # 这里采用最简：走 HTTP 内部逻辑
    # 为了不重复大量代码，构造一个最小 resume 触发

    # 复用 resume_novel 的逻辑：直接调用它自己的后台任务
    # 或者更简单：让客户端再调一次 /resume

    # 这里返回 task_id = None，让客户端调 /resume
    return {
        "status": "rerun_ready",
        "novel_id": novel_id,
        "volume_num": volume_num,
        "chapter_num": chapter_num,
        "deleted_rows": len(existing_scenes),
        "backup_files": backup_files,
        "next_action": f"POST /api/v1/novel/resume with {{'novel_id': '{novel_id}'}}",
        "message": "DB and files cleared. Call /resume to regenerate this chapter."
    }