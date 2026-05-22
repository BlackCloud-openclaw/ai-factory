# src/orchestrator/nodes.py
import uuid
import time
import json
import re
import asyncio
from typing import Any, Tuple, List, Dict
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
from src.writing.causality.initializer import ensure_core_predicates
import logging
from src.writing.services import SceneCompletionService, SceneCompletionCommand
from src.orchestrator.state_patch import StatePatch, WorkflowPhase
from src.orchestrator.phase_resolver import WorkflowPhaseResolver
from src.writing.services.scene_planning import ScenePlanningService
from src.writing.services.models import ScenePlanningCommand
from src.writing.services.writing import WritingService
from src.writing.services.models import WritingCommand
from src.writing.services.chapter_transition import ChapterTransitionService, ChapterTransitionCommand

logger = setup_logging("orchestrator.nodes")

_memory_agent = MemoryAgent()

async def _load_scene_plans_from_db(
    pool, novel_id: str, volume_num: int, chapter_num: int
) -> Tuple[List[Dict[str, Any]], int]:
    """
    从 scene_execution_units 表加载指定章节的场景计划列表。
    返回 (scene_plan_list, total_scenes)
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT plan_json, scene_index, status
            FROM scene_execution_units
            WHERE novel_id = $1 AND volume_num = $2 AND chapter_num = $3
            ORDER BY scene_index ASC
            """,
            novel_id, volume_num, chapter_num
        )
        scene_plans = []
        for row in rows:
            plan = json.loads(row["plan_json"])
            # 可选：添加状态字段用于调试，但不影响计划内容
            # plan["_status"] = row["status"]
            scene_plans.append(plan)
        return scene_plans, len(scene_plans)

async def _get_succeeded_scenes_count(
    pool, novel_id: str, volume_num: int, chapter_num: int
) -> int:
    """返回指定章节中状态为 'succeeded' 的场景数量"""
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM scene_execution_units
            WHERE novel_id = $1 AND volume_num = $2 AND chapter_num = $3
            AND status = 'succeeded'
            """,
            novel_id, volume_num, chapter_num
        )
        return count or 0

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

async def plan_node(state: AgentState) -> dict:
    logger.info(f"plan_node called with chapter={state.current_chapter}, task_type={state.task_type}")

    cmd = ScenePlanningCommand(
        novel_id=state.novel_id,
        volume=state.current_volume,
        chapter=state.current_chapter,
        task_type=state.task_type,
        outline=state.outline,
        current_state=state.current_state,
        user_input=state.user_input,
        resume=state.resume,
        total_chapters_in_volume=getattr(state, 'total_chapters_in_volume', 0),
    )
    result = await ScenePlanningService.execute(cmd)
    if result.error:
        logger.error(f"ScenePlanningService failed: {result.error}")
        return StatePatch(error=result.error).to_dict()
    return result.state_patch.to_dict()

async def writer_node(state: AgentState) -> dict:
    """写作节点：调用 WritingService 生成场景"""
    scene_plan_list = state.scene_plan_list
    current_idx = state.current_scene_index if state.current_scene_index is not None else 0

    if current_idx >= len(scene_plan_list):
        logger.error(f"writer_node: invalid scene index {current_idx} (list length {len(scene_plan_list)})")
        return StatePatch(error="Invalid scene index").to_dict()

    current_scene_plan = scene_plan_list[current_idx]
    state.scene_plan = current_scene_plan

    # 更新场景执行单元状态为 running（可以在 service 外做，因为不是事务核心）
    if state.novel_id and state.task_type == "scene_plan":
        await _update_scene_unit_status(
            state.novel_id, state.current_volume, state.current_chapter, current_idx, "running"
        )

    # 调用写作服务
    cmd = WritingCommand(
        novel_id=state.novel_id,
        volume=state.current_volume,
        chapter=state.current_chapter,
        scene_idx=current_idx,
        scene_plan=current_scene_plan,
        current_state=state.current_state,
        writing_feedback=getattr(state, "writing_feedback", ""),
    )
    result = await WritingService.execute(cmd)

    if result.error:
        logger.error(f"WritingService failed: {result.error}")
        return StatePatch(error=result.error).to_dict()

    # 返回 patch（LangGraph 会自动合并）
    return result.state_patch.to_dict()

logger = logging.getLogger(__name__)

async def validate_node(state: AgentState) -> dict:
    """
    验证场景并推进工作流。
    重构后：验证 -> 处理重试 -> 调用 SceneCompletionService -> 返回 patch
    """
    # 确保验证模式为小说模式
    state.validation_mode = "novel"
    
    # 1. 验证（使用现有 ValidatorAgent）
    validator = ValidatorAgent()
    updates = await validator.run(state)
    validation_result = updates.get("validation_result", {})
    passed = validation_result.get("passed", False)
    should_retry = validation_result.get("should_retry", False)

    # 2. 失败重试逻辑
    if not passed and should_retry:
        retry_count = state.retry_count + 1
        if retry_count < state.max_retries_per_subtask:
            logger.info(f"Scene {state.current_scene_index} validation failed, retrying ({retry_count}/{state.max_retries_per_subtask})")
            return StatePatch(
                validation_result=validation_result,
                retry_count=retry_count,
                needs_retry=True,
                writing_feedback=validation_result.get("feedback", ""),
                phase=WorkflowPhase.WRITING,
            ).to_dict()
        else:
            # 超过重试次数，跳过该场景
            logger.warning(f"Scene {state.current_scene_index} validation failed after {retry_count} retries, skipping")
            await _skip_scene(state)
            return StatePatch(
                current_scene_index=state.current_scene_index + 1,
                retry_count=0,
                phase=WorkflowPhase.WRITING,
            ).to_dict()

    # 3. 验证通过，调用服务
    parsed_output = validation_result.get("parsed_output", {})
    if parsed_output:
        events = parsed_output.get("events", [])
        for evt in events:
            if evt.get("type") == "discovery" and "importance" in evt:
                imp = evt["importance"]
                if isinstance(imp, int):
                    if imp >= 5:
                        evt["importance"] = "critical"
                    elif imp >= 3:
                        evt["importance"] = "high"
                    elif imp >= 1:
                        evt["importance"] = "normal"
                    else:
                        evt["importance"] = "low"
                elif isinstance(imp, float):
                    evt["importance"] = "critical" if imp >= 5 else "high" if imp >= 3 else "normal" if imp >= 1 else "low"
                elif isinstance(imp, bool):
                    evt["importance"] = "critical" if imp else "low"
        parsed_output["events"] = events
        validation_result["parsed_output"] = parsed_output
    
    cmd = SceneCompletionCommand(
        novel_id=state.novel_id,
        volume=state.current_volume,
        chapter=state.current_chapter,
        scene_idx=state.current_scene_index,
        total_scenes=state.total_scenes_in_chapter,
        current_world_state=state.current_state,
        parsed_output=parsed_output,
        scene_plan=state.scene_plan,
    )
    result = await SceneCompletionService.execute(cmd)
    logger.info(f"validate_node: service returned chapter_finished={result.chapter_finished}")

    # 4. 保存场景到文件（如果存在正文）
    if parsed_output.get("scene_text"):
        await _save_scene_to_file(state, parsed_output["scene_text"])

    # 如果章节完成，执行切换
    if result.chapter_finished:
        # 忽略场景完成返回的 patch（除了可能需要保留的 state 等，但切换会清空场景计划，所以直接返回切换 patch）
        transition_cmd = ChapterTransitionCommand(...)
        transition_result = await ChapterTransitionService.execute(transition_cmd)
        # 仍需保留世界状态和验证结果（可选）
        transition_result.state_patch.current_state = result.state_patch.current_state
        transition_result.state_patch.validation_result = result.state_patch.validation_result
        return transition_result.state_patch.to_dict()
    else:
        return result.state_patch.to_dict()

async def _skip_scene(state: AgentState):
    """跳过当前场景：更新 writing_progress 和 scene_execution_units 状态"""
    pool = get_db_pool()
    if not pool:
        return
    async with pool.acquire() as conn:
        async with conn.transaction():
            # 更新 writing_progress
            new_scene_idx = state.current_scene_index + 1
            await conn.execute(
                """
                INSERT INTO writing_progress (project_id, current_volume, current_chapter, current_scene, chapter_completed, last_updated)
                VALUES ($1, $2, $3, $4, false, NOW())
                ON CONFLICT (project_id) DO UPDATE SET
                    current_scene = EXCLUDED.current_scene,
                    last_updated = NOW()
                """,
                state.novel_id, state.current_volume, state.current_chapter, new_scene_idx
            )
            # 标记 scene_execution_units 为 skipped
            await conn.execute(
                """
                UPDATE scene_execution_units
                SET status = 'skipped', completed_at = NOW(), updated_at = NOW()
                WHERE novel_id = $1 AND volume_num = $2 AND chapter_num = $3 AND scene_index = $4
                """,
                state.novel_id, state.current_volume, state.current_chapter, state.current_scene_index
            )

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