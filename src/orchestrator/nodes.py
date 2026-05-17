# src/orchestrator/nodes.py
import uuid
import time
import json
import re
import asyncio
from typing import Any
from pathlib import Path

from src.orchestrator.state import AgentState
from src.agents.research import ResearchAgent
from src.agents.executor import ExecutorAgent
from src.agents.memory import MemoryAgent
from src.agents.planner import PlannerAgent
from src.agents.validator import ValidatorAgent
from src.agents.writer import WritingAgent
from src.common.logging import setup_logging
from src.db import get_db_pool
from src.writing.summarizer import generate_chapter_summary
from src.writing.state_compressor import compress_current_state
from src.writing.world_state import WorldState
from src.writing.delta import StateDelta
from src.writing.event_store import NarrativeEventStore
from src.writing.snapshot import SnapshotManager
from src.writing.validators import validate_all
from src.writing.prompt_firewall import PromptFirewall
from src.writing.context_compiler import ContextCompiler
from src.writing.voiceprint import VoiceprintRegistry
from src.db.pool import update_progress_scene, update_progress_chapter, update_progress_volume

logger = setup_logging("orchestrator.nodes")

_memory_agent = MemoryAgent()


def get_memory_agent() -> MemoryAgent:
    return _memory_agent


# ====== Helper functions ======
def _keyword_analyze(user_input: str) -> tuple[str, list[str]]:
    lower = user_input.lower()
    if any(kw in lower for kw in ["write", "code", "implement", "function", "class", "create"]):
        intent = "code_generation"
    elif any(kw in lower for kw in ["explain", "what is", "how does", "tell me", "research", "knowledge"]):
        intent = "research"
    else:
        intent = "general_chat"
    return intent, [user_input]


def _is_complex_task(user_input: str) -> bool:
    return len(user_input) > 200


def _build_research_summary(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No research results available."
    return "\n\n".join(f"[{r.get('source','unknown')}]: {r.get('summary', r.get('content','No content'))}" for r in results)


async def _save_scene_to_file(state: AgentState, raw_text: str) -> None:
    if not state.novel_id or not raw_text:
        return
    try:
        novel_data_dir = Path(f"data/novels/{state.novel_id}")
        volumes_dir = novel_data_dir / f"vol_{state.current_volume:03d}"
        volumes_dir.mkdir(parents=True, exist_ok=True)
        chapter_file = volumes_dir / f"chap_{state.current_chapter:03d}.txt"
        mode = "a" if chapter_file.exists() else "w"
        with open(chapter_file, mode, encoding="utf-8") as f:
            if mode == "a":
                f.write("\n\n<!-- scene break -->\n\n")
            f.write(raw_text)
        logger.info(f"Saved scene to {chapter_file} (mode={mode}, length={len(raw_text)})")
    except Exception as e:
        logger.error(f"Failed to save scene: {e}")


# ====== 场景执行单元辅助函数 ======
async def _update_scene_unit_status(
    novel_id: str,
    volume: int,
    chapter: int,
    scene_index: int,
    status: str,
    error_msg: str = None,
    actual_state_delta: dict = None,
) -> None:
    """更新 scene_execution_units 表的状态和可选字段"""
    pool = get_db_pool()
    if not pool:
        return
    try:
        async with pool.acquire() as conn:
            if status == "running":
                await conn.execute("""
                    UPDATE scene_execution_units
                    SET status = 'running', started_at = NOW(), updated_at = NOW()
                    WHERE novel_id = $1 AND volume_num = $2 AND chapter_num = $3 AND scene_index = $4
                    AND status = 'pending'
                """, novel_id, volume, chapter, scene_index)
                logger.info(f"Scene {scene_index} status -> running")
            elif status == "succeeded":
                actual_json = json.dumps(actual_state_delta) if actual_state_delta else None
                await conn.execute("""
                    UPDATE scene_execution_units
                    SET status = 'succeeded', actual_state_delta = $5, completed_at = NOW(), updated_at = NOW()
                    WHERE novel_id = $1 AND volume_num = $2 AND chapter_num = $3 AND scene_index = $4
                """, novel_id, volume, chapter, scene_index, actual_json)
                logger.info(f"Scene {scene_index} status -> succeeded")
            elif status == "failed":
                await conn.execute("""
                    UPDATE scene_execution_units
                    SET status = 'failed', error_message = $5, completed_at = NOW(), updated_at = NOW()
                    WHERE novel_id = $1 AND volume_num = $2 AND chapter_num = $3 AND scene_index = $4
                """, novel_id, volume, chapter, scene_index, error_msg)
                logger.info(f"Scene {scene_index} status -> failed")
            elif status == "skipped":
                await conn.execute("""
                    UPDATE scene_execution_units
                    SET status = 'skipped', error_message = $5, completed_at = NOW(), updated_at = NOW()
                    WHERE novel_id = $1 AND volume_num = $2 AND chapter_num = $3 AND scene_index = $4
                """, novel_id, volume, chapter, scene_index, error_msg)
                logger.info(f"Scene {scene_index} status -> skipped")
            elif status == "increment_retry":
                await conn.execute("""
                    UPDATE scene_execution_units
                    SET retry_count = retry_count + 1, updated_at = NOW()
                    WHERE novel_id = $1 AND volume_num = $2 AND chapter_num = $3 AND scene_index = $4
                """, novel_id, volume, chapter, scene_index)
                logger.info(f"Incremented retry_count for scene {scene_index}")
    except Exception as e:
        logger.error(f"Failed to update scene unit status {status}: {e}", exc_info=True)


async def _persist_scene_plans(state: AgentState, scenes: list) -> None:
    """将场景计划持久化到 scene_execution_units"""
    if not state.novel_id:
        return
    pool = get_db_pool()
    if not pool:
        return
    try:
        async with pool.acquire() as conn:
            for idx, scene in enumerate(scenes):
                scene_index = idx
                plan_json = json.dumps(scene, ensure_ascii=False)
                planned_state_delta = json.dumps(scene.get("state_delta", {}), ensure_ascii=False) if scene.get("state_delta") else None
                await conn.execute("""
                    INSERT INTO scene_execution_units 
                    (novel_id, volume_num, chapter_num, scene_index, status, plan_json, planned_state_delta, retry_count, max_retries, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, 'pending', $5, $6, 0, 2, NOW(), NOW())
                    ON CONFLICT (novel_id, volume_num, chapter_num, scene_index)
                    DO UPDATE SET
                        plan_json = EXCLUDED.plan_json,
                        planned_state_delta = EXCLUDED.planned_state_delta,
                        status = 'pending',
                        retry_count = 0,
                        updated_at = NOW()
                """, state.novel_id, state.current_volume, state.current_chapter, scene_index, plan_json, planned_state_delta)
        logger.info(f"Persisted {len(scenes)} scenes to scene_execution_units for chapter {state.current_chapter}")
    except Exception as e:
        logger.error(f"Failed to persist scene_execution_units: {e}", exc_info=True)


# ====== Node functions ======
async def load_memory_node(state: AgentState) -> dict[str, Any]:
    return await _memory_agent.run(state)


async def save_memory_node(state: AgentState) -> dict[str, Any]:
    logger.info(f"Save memory for {state.novel_id}, outline exists: {state.outline is not None}")
    
    if state.outline and state.novel_id:
        pool = get_db_pool()
        if pool:           
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with pool.acquire() as conn:
                        row = await conn.fetchrow("SELECT revision FROM novels WHERE novel_id = $1", state.novel_id)
                        if row is None:
                            await conn.execute("""
                                INSERT INTO novels (novel_id, title, outline, current_volume, current_chapter, current_scene_index, revision, created_at, updated_at)
                                VALUES ($1, $2, $3, $4, $5, $6, 1, NOW(), NOW())
                            """, state.novel_id,
                                state.outline.get("title", "Untitled") if isinstance(state.outline, dict) else "Untitled",
                                json.dumps(state.outline),
                                state.current_volume,
                                state.current_chapter,
                                state.current_scene_index if state.current_scene_index is not None else 0)
                            logger.info(f"✅ Inserted novel record for {state.novel_id}")
                            break
                        else:
                            old_revision = row["revision"]
                            result = await conn.execute("""
                                UPDATE novels 
                                SET outline = $2,
                                    current_volume = $3,
                                    current_chapter = $4,
                                    current_scene_index = $5,
                                    revision = revision + 1,
                                    updated_at = NOW()
                                WHERE novel_id = $1 AND revision = $6
                            """, state.novel_id, 
                                json.dumps(state.outline),
                                state.current_volume,
                                state.current_chapter,
                                state.current_scene_index if state.current_scene_index is not None else 0,
                                old_revision)
                            if result == "UPDATE 0":
                                logger.warning(f"Optimistic lock conflict for {state.novel_id}, retry {attempt+1}/{max_retries}")
                                await asyncio.sleep(0.1)
                                continue
                            else:
                                logger.info(f"✅ Updated outline for {state.novel_id}")
                                break
                except Exception as e:
                    logger.error(f"Failed to save outline: {e}", exc_info=True)
                    break
            else:
                logger.error(f"Failed to save outline after {max_retries} retries")
            
            # 强制同步 writing_progress
            if state.novel_id and state.task_type == "scene_plan":
                from src.db.pool import init_writing_progress
                await init_writing_progress(
                    state.novel_id,
                    volume=state.current_volume,
                    chapter=state.current_chapter,
                    scene=state.current_scene_index if state.current_scene_index is not None else 0,
                    chapter_completed=False
                )
                logger.info(f"Synced writing_progress for {state.novel_id} (volume={state.current_volume}, chapter={state.current_chapter}, scene={state.current_scene_index})")
    else:
        logger.warning(f"state.outline is empty for {state.novel_id}, cannot save")
    
    return {"metadata": state.metadata, "novel_id": state.novel_id}


async def analyze_node(state: AgentState) -> dict[str, Any]:
    intent, subtasks = _keyword_analyze(state.user_input)
    return {"intent": intent, "subtasks": subtasks, "is_complex": _is_complex_task(state.user_input), "current_node": "analyze"}


async def plan_node(state: AgentState) -> dict[str, Any]:
    logger.info(f"plan_node called with chapter={state.current_chapter}, task_type={state.task_type}")
    
    updates = {}
    pool = get_db_pool()
    
    # 重新加载 outline（如果需要）
    if state.outline is None and state.novel_id and pool:
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow("SELECT outline FROM novels WHERE novel_id = $1", state.novel_id)
                if row and row["outline"]:
                    state.outline = json.loads(row["outline"])
                    logger.info(f"✅ Reloaded outline for {state.novel_id}")
                    if state.outline and "volumes" in state.outline:
                        volumes = state.outline["volumes"]
                        vol_idx = state.current_volume - 1
                        if 0 <= vol_idx < len(volumes):
                            total_chapters = len(volumes[vol_idx].get("chapters", []))
                            state.total_chapters_in_volume = total_chapters
                            updates["total_chapters_in_volume"] = total_chapters
                            logger.info(f"Set total_chapters_in_volume={total_chapters}")
                else:
                    logger.warning(f"⚠️ No outline found in DB for {state.novel_id}")
        except Exception as e:
            logger.error(f"Failed to reload outline: {e}")
    
    # 上下文注入
    if state.task_type == "scene_plan" and state.novel_id:
        try:
            world = WorldState.from_dict(state.current_state) if state.current_state else WorldState()
            compiler = ContextCompiler()
            current_volume_outline = None
            if state.outline and "volumes" in state.outline:
                volumes = state.outline.get("volumes", [])
                vol_idx = state.current_volume - 1
                if 0 <= vol_idx < len(volumes):
                    current_volume_outline = volumes[vol_idx]
            compiled = compiler.compile_for_planner(
                world, state.current_volume, state.current_chapter,
                current_volume_outline or state.outline or {}
            )
            state.metadata["compiled_context"] = compiled
        except Exception as e:
            logger.warning(f"Context compile failed: {e}")
    
    # 调用 PlannerAgent
    planner = PlannerAgent()
    planner_updates = await planner.run(state)
    updates.update(planner_updates)
    
    # 处理场景计划
    if state.task_type == "scene_plan" and updates.get("scene_plan"):
        scene_plan_data = updates["scene_plan"]
        if isinstance(scene_plan_data, dict):
            scenes = scene_plan_data.get("scenes", [])
        elif isinstance(scene_plan_data, list):
            scenes = scene_plan_data
        else:
            scenes = []
        
        logger.info(f"plan_node: extracted {len(scenes)} scenes from planner response")
        if scenes:
            for i, scene in enumerate(scenes):
                if "must_events" not in scene or not scene["must_events"]:
                    scene["must_events"] = [f"推进主线剧情（场景{i+1}）"]
                else:
                    scene["must_events"] = [e for e in scene["must_events"] if "推进主线剧情" not in e]
                if "state_delta" not in scene:
                    scene["state_delta"] = {"events": []}
                if "depends_on" not in scene:
                    scene["depends_on"] = []
                if "scene_id" not in scene:
                    scene["scene_id"] = i + 1
            
            updates["scene_plan_list"] = scenes
            updates["total_scenes_in_chapter"] = len(scenes)
            updates["scene_plan"] = scenes[0] if scenes else {}
            state.scene_plan_list = scenes
            state.total_scenes_in_chapter = len(scenes)
            state.scene_plan = scenes[0] if scenes else {}
            logger.info(f"plan_node: set state.scene_plan (exists={state.scene_plan is not None}, scenes={len(scenes)})")
            
            state.metadata["scene_plan_list"] = scenes
            state.metadata["total_scenes_in_chapter"] = len(scenes)
            state.metadata["current_scene_index"] = 0
            state.metadata["current_scene_plan"] = scenes[0] if scenes else None
            logger.info(f"plan_node: stored scene plans in metadata, count={len(scenes)}")
            
            # 持久化到 scene_execution_units
            await _persist_scene_plans(state, scenes)
        else:
            logger.warning("plan_node: no scenes extracted from planner response")
    
    # 确保 total_chapters_in_volume
    if state.task_type == "scene_plan" and state.novel_id:
        total_chapters = getattr(state, 'total_chapters_in_volume', 0)
        if total_chapters == 0 and state.outline and "volumes" in state.outline:
            volumes = state.outline["volumes"]
            vol_idx = state.current_volume - 1
            if 0 <= vol_idx < len(volumes):
                total_chapters = len(volumes[vol_idx].get("chapters", []))
        if total_chapters == 0:
            total_chapters = 10
            logger.warning(f"Could not determine total_chapters_in_volume, using default {total_chapters}")
        updates["total_chapters_in_volume"] = total_chapters
    
    logger.info(f"plan_node returning: scene_plan in state = {state.scene_plan is not None}")
    logger.info(f"plan_node returning: scene_plan in updates = {updates.get('scene_plan') is not None}")
    updates["metadata"] = state.metadata.copy()
    updates["novel_id"] = state.novel_id
    return updates


async def writer_node(state: AgentState) -> dict[str, Any]:
    if state.total_scenes_in_chapter == 0 and state.metadata.get("total_scenes_in_chapter"):
        state.total_scenes_in_chapter = state.metadata["total_scenes_in_chapter"]
    if state.current_scene_index is None and state.metadata.get("current_scene_index") is not None:
        state.current_scene_index = state.metadata["current_scene_index"]
    
    scene_plan_list = state.metadata.get("scene_plan_list", [])
    current_idx = state.metadata.get("current_scene_index", 0)
    if current_idx < len(scene_plan_list):
        current_scene_plan = scene_plan_list[current_idx]
    else:
        current_scene_plan = None
    
    if current_scene_plan is None:
        logger.error(f"writer_node: no scene plan for index {current_idx}, list length {len(scene_plan_list)}")
        return {"scene_text": "", "final_answer": "", "current_node": "writer"}
    
    state.metadata["current_scene_plan"] = current_scene_plan
    state.scene_plan = current_scene_plan
    
    if state.scene_plan is None:
        logger.error("writer_node: state.scene_plan is None, cannot write")
        return {"scene_text": "", "final_answer": "", "current_node": "writer"}
    
    # 注入验证反馈
    if hasattr(state, 'writing_feedback') and state.writing_feedback:
        state.metadata["writing_feedback"] = state.writing_feedback
    else:
        state.metadata.pop("writing_feedback", None)
    
    # 更新场景执行单元状态为 running
    if state.novel_id and state.task_type == "scene_plan" and current_scene_plan:
        await _update_scene_unit_status(state.novel_id, state.current_volume, state.current_chapter, current_idx, "running")
    
    writer = WritingAgent()
    result = await writer.run(state)
    raw_json = result.get("scene_text", "")
    
    # 提取 scene_text
    clean_text = ""
    json_match = re.search(r'\{.*\}', raw_json, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            clean_text = data.get("scene_text", "")
        except:
            clean_text = raw_json
    else:
        clean_text = raw_json
    
    return {
        "scene_text": raw_json,
        "final_answer": clean_text,
        "current_node": "writer",
        "deviation_detected": result.get("deviation_detected", False),
        "missing_goal_keywords": result.get("missing_goal_keywords", []),
        "missing_conflict_keywords": result.get("missing_conflict_keywords", []),
        "metadata": state.metadata,
        "novel_id": state.novel_id,
    }


async def validate_node(state: AgentState) -> dict[str, Any]:
    if state.total_scenes_in_chapter == 0 and state.metadata.get("total_scenes_in_chapter"):
        state.total_scenes_in_chapter = state.metadata["total_scenes_in_chapter"]
    
    scene_plan_list = state.metadata.get("scene_plan_list", [])
    current_scene_index = state.metadata.get("current_scene_index", 0)
    total_scenes = state.metadata.get("total_scenes_in_chapter", 0)
    
    if current_scene_index < len(scene_plan_list):
        current_scene_plan = scene_plan_list[current_scene_index]
    else:
        current_scene_plan = None
        logger.warning(f"validate_node: invalid scene index {current_scene_index} for list length {len(scene_plan_list)}")
    
    state.scene_plan = current_scene_plan if current_scene_plan else {}
    mode = "novel" if state.scene_text else "code"
    state.validation_mode = mode
    validator = ValidatorAgent()
    updates = await validator.run(state)
    
    validation_result = updates.get("validation_result", {})
    passed = validation_result.get("passed", False)
    should_retry = validation_result.get("should_retry", False)
    need_semantic = validation_result.get("need_semantic", False)
    
    # 异步语义验证
    if need_semantic and state.scene_text and state.scene_plan:
        from src.writing.validators import validate_semantic
        semantic_passed, semantic_error = await validate_semantic(
            state.scene_text,
            {"must_events": state.scene_plan.get("must_events", [])}
        )
        if not semantic_passed:
            passed = False
            should_retry = True
            validation_result["passed"] = False
            validation_result["feedback"] = semantic_error
            validation_result["should_retry"] = True
            logger.warning(f"Semantic validation failed: {semantic_error}")
    
    # 处理重试逻辑
    if not passed and should_retry:
        retry_count = getattr(state, 'retry_count', 0) + 1
        max_retries = getattr(state, 'max_retries_per_subtask', 2)
        logger.warning(f"Validation failed: {validation_result.get('feedback', '')}")
        
        if retry_count < max_retries:
            logger.info(f"Retrying scene {current_scene_index + 1} (retry {retry_count}/{max_retries})")
            if state.novel_id and state.task_type == "scene_plan":
                await _update_scene_unit_status(state.novel_id, state.current_volume, state.current_chapter, current_scene_index, "increment_retry")
            return {
                "validation_result": validation_result,
                "error": validation_result.get("feedback"),
                "retry_count": retry_count,
                "needs_retry": True,
                "writing_feedback": validation_result.get("feedback", ""),
                "metadata": state.metadata,
                "novel_id": state.novel_id,
            }
        else:
            logger.error(f"Validation failed after {max_retries} retries, skipping scene")
            if state.novel_id and state.task_type == "scene_plan":
                await _update_scene_unit_status(state.novel_id, state.current_volume, state.current_chapter, current_scene_index,
                                                "skipped", validation_result.get("feedback", "Validation failed after retries"))
            
            new_idx = current_scene_index + 1
            if state.novel_id and state.task_type == "scene_plan":
                try:
                    await update_progress_scene(state.novel_id, new_idx, chapter_completed=False)
                except Exception as e:
                    logger.error(f"Failed to update writing_progress for skipped scene: {e}", exc_info=True)
            
            base_updates = {
                "validation_result": validation_result,
                "error": validation_result.get("feedback"),
                "current_scene_index": new_idx,
                "retry_count": 0,
                "needs_retry": False,
                "novel_id": state.novel_id,
            }
            state.metadata["current_scene_index"] = new_idx
            if new_idx < len(scene_plan_list):
                next_scene_plan = scene_plan_list[new_idx]
                state.metadata["current_scene_plan"] = next_scene_plan
                base_updates["scene_plan"] = next_scene_plan
                logger.info(f"Skipped scene, updated scene_plan for next scene (index {new_idx})")
            
            if total_scenes > 0 and new_idx >= total_scenes:
                logger.info(f"Chapter {state.current_chapter} completed (with skipped scenes), advancing to next chapter")
                new_chapter = state.current_chapter + 1
                base_updates["current_chapter"] = new_chapter
                base_updates["current_scene_index"] = 0
                base_updates["scene_plan_list"] = []
                base_updates["total_scenes_in_chapter"] = 0
                base_updates["_chapter_finished"] = True
                base_updates["scene_plan"] = None
                state.metadata["scene_plan_list"] = []
                state.metadata["total_scenes_in_chapter"] = 0
                state.metadata["current_scene_index"] = 0
                state.metadata["current_scene_plan"] = None
                if state.novel_id:
                    try:
                        await update_progress_chapter(state.novel_id, new_chapter)
                    except Exception as e:
                        logger.error(f"Failed to update writing_progress for chapter (skip branch): {e}", exc_info=True)
                total_chapters = getattr(state, "total_chapters_in_volume", 0)
                if total_chapters > 0 and new_chapter > total_chapters:
                    new_vol = state.current_volume + 1
                    base_updates["current_volume"] = new_vol
                    base_updates["current_chapter"] = 1
                    logger.info(f"Volume {state.current_volume} completed! Moving to volume {new_vol}")
                    if state.novel_id:
                        try:
                            await update_progress_volume(state.novel_id, new_vol)
                        except Exception as e:
                            logger.error(f"Failed to update writing_progress for volume: {e}", exc_info=True)
            base_updates["metadata"] = state.metadata
            return base_updates
    
    # 致命错误
    if not passed:
        logger.error(f"Validation failed (fatal): {validation_result.get('feedback', '')}")
        if state.novel_id and state.task_type == "scene_plan":
            await _update_scene_unit_status(state.novel_id, state.current_volume, state.current_chapter, current_scene_index,
                                            "failed", validation_result.get("feedback", "Fatal validation error"))
        return {
            "validation_result": validation_result,
            "error": validation_result.get("feedback"),
            "needs_retry": False,
            "metadata": state.metadata,
            "novel_id": state.novel_id,
        }
    
    # 验证通过，应用状态变更和存储事件
    parsed_output = validation_result.get("parsed_output") if validation_result else None
    
    if state.task_type == "scene_plan" and state.scene_text:
        pool = get_db_pool()
        if pool:
            try:
                current_world = WorldState.from_dict(state.current_state) if state.current_state else WorldState()
                new_world = current_world
                events_applied = False
                events = []
                
                if parsed_output:
                    events_data = parsed_output.get("events", [])
                    if events_data:
                        from src.writing.events import event_from_dict
                        for evt_data in events_data:
                            evt_type = evt_data.get("type")
                            if evt_type:
                                evt = event_from_dict(evt_type, evt_data)
                                if evt:
                                    events.append(evt)
                        if events:
                            delta = StateDelta(events=events)
                            new_world = delta.apply_to(current_world)
                            events_applied = True
                            logger.info(f"Applied {len(events)} events from Writer output")
                
                if not events_applied:
                    planned_delta_dict = state.scene_plan.get("state_delta", {}) if state.scene_plan else {}
                    if planned_delta_dict:
                        delta = StateDelta.from_dict(planned_delta_dict)
                        new_world = delta.apply_to(current_world)
                        events_applied = True
                        logger.info("Applied state_delta from Planner (fallback)")
                
                if events_applied and pool:
                    if parsed_output and parsed_output.get("events"):
                        event_store = NarrativeEventStore(pool)
                        for evt in events:
                            await event_store.append_event(
                                state.novel_id, evt,
                                state.current_volume, state.current_chapter,
                                current_scene_index
                            )
                        last_id = await event_store.get_last_event_id(state.novel_id)
                        chapter_finished = total_scenes > 0 and (current_scene_index + 1 >= total_scenes)
                        if chapter_finished:
                            snap_mgr = SnapshotManager(pool)
                            await snap_mgr.save_snapshot(
                                state.novel_id, new_world, last_id,
                                state.current_volume, state.current_chapter
                            )
                    updates["current_state"] = new_world.to_dict()
                    
                    # 更新场景执行单元状态为 succeeded，并保存实际状态增量
                    if state.novel_id and state.task_type == "scene_plan" and current_scene_plan:
                        actual_events = parsed_output.get("events", []) if parsed_output else []
                        actual_state_delta = {"events": actual_events} if actual_events else None
                        await _update_scene_unit_status(state.novel_id, state.current_volume, state.current_chapter, current_scene_index,
                                                        "succeeded", actual_state_delta=actual_state_delta)
                    logger.info("World state updated and events stored")
            except Exception as e:
                logger.error(f"State delta application error: {e}", exc_info=True)
        
        # 更新场景索引
        new_idx = current_scene_index + 1
        state.metadata["current_scene_index"] = new_idx
        updates["current_scene_index"] = new_idx
        updates["retry_count"] = 0
        updates["total_scenes_in_chapter"] = total_scenes
        
        if state.novel_id:
            try:
                await update_progress_scene(
                    state.novel_id,
                    scene_index=new_idx,
                    chapter_completed=(total_scenes > 0 and new_idx >= total_scenes)
                )
            except Exception as e:
                logger.error(f"Failed to update writing_progress for scene: {e}", exc_info=True)
        
        if new_idx < len(scene_plan_list):
            next_scene_plan = scene_plan_list[new_idx]
            state.metadata["current_scene_plan"] = next_scene_plan
            updates["scene_plan"] = next_scene_plan
            logger.info(f"Updated scene_plan for next scene (index {new_idx}): goal={next_scene_plan.get('goal', '')[:60]}")
        
        if state.novel_id and state.final_answer:
            await _save_scene_to_file(state, state.final_answer)
        
        logger.info(f"validate_node: total_scenes={total_scenes}, current_idx={current_scene_index}, new_idx={new_idx}, current_chapter={state.current_chapter}")
        
        if total_scenes > 0 and new_idx >= total_scenes:
            new_chapter = state.current_chapter + 1
            logger.info(f"✅ Chapter {state.current_chapter} completed! Advancing to chapter {new_chapter}")
            updates["current_chapter"] = new_chapter
            updates["current_scene_index"] = 0
            updates["scene_plan_list"] = []
            updates["total_scenes_in_chapter"] = 0
            updates["_chapter_finished"] = True
            updates["scene_plan"] = None
            state.metadata["scene_plan_list"] = []
            state.metadata["total_scenes_in_chapter"] = 0
            state.metadata["current_scene_index"] = 0
            state.metadata["current_scene_plan"] = None
            if state.novel_id:
                try:
                    await update_progress_chapter(state.novel_id, new_chapter)
                except Exception as e:
                    logger.error(f"Failed to update writing_progress for chapter: {e}", exc_info=True)
            total_chapters = getattr(state, "total_chapters_in_volume", 0)
            if total_chapters > 0 and new_chapter > total_chapters:
                new_vol = state.current_volume + 1
                updates["current_volume"] = new_vol
                updates["current_chapter"] = 1
                logger.info(f"📚 Volume {state.current_volume} completed! Moving to volume {new_vol}")
                if state.novel_id:
                    try:
                        await update_progress_volume(state.novel_id, new_vol)
                    except Exception as e:
                        logger.error(f"Failed to update writing_progress for volume: {e}", exc_info=True)
        else:
            logger.info(f"Continuing with next scene of chapter {state.current_chapter}")
    
    updates["metadata"] = state.metadata
    updates["novel_id"] = state.novel_id
    return updates


async def research_node(state: AgentState) -> dict[str, Any]:
    ra = ResearchAgent()
    result = await ra.run(state)
    return {"research_results": result.get("research_results", []), "sources": result.get("sources", []), "current_node": "research"}


async def code_node(state: AgentState) -> dict[str, Any]:
    ea = ExecutorAgent()
    updates = await ea.run(state)
    return {"code_generated": updates.get("code_generated", ""), "code_file_path": updates.get("code_file_path", ""), "execution_result": updates.get("execution_result"), "current_node": "code"}


async def scheduler_node(state: AgentState) -> dict[str, Any]:
    return {"plan_status": "no_plan", "subtask_results": {}, "current_node": "scheduler"}


def advance_subtask_node(state: AgentState) -> dict[str, Any]:
    return {"subtasks": []}


async def tool_node_v2(state: AgentState) -> dict[str, Any]:
    return {"pending_tool_calls": [], "tool_results": [], "current_node": "tool_node"}