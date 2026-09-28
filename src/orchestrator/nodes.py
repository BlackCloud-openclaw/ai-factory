# src/orchestrator/nodes.py
"""
LangGraph 节点实现 - 薄编排层

职责：
- 调用 Service 执行业务事务
- 返回 StatePatch 更新状态
- 不包含业务逻辑、数据库操作、复杂条件判断
"""

import uuid
import time
import json
import re
import asyncio
from typing import Any, Tuple, List, Dict, Optional
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
from src.writing.delta import StateDelta
from src.writing.event_store import NarrativeEventStore
from src.writing.snapshot_manager import SnapshotManager
from src.writing.context_compiler import ContextCompiler
from src.writing.causality.initializer import ensure_core_predicates
from src.writing.services import SceneCompletionService, SceneCompletionCommand
from src.orchestrator.state_patch import StatePatch, WorkflowPhase
from src.orchestrator.phase_resolver import WorkflowPhaseResolver
from src.writing.services.scene_planning import ScenePlanningService
from src.writing.services.models import ScenePlanningCommand
from src.writing.services.writing import WritingService
from src.writing.services.models import WritingCommand
from src.writing.services.chapter_transition import ChapterTransitionService, ChapterTransitionCommand
from src.writing.narrative_entropy import NarrativeEntropyCalculator
from src.writing.memory_hierarchy import CompressedState
from src.orchestrator.audit import audit_state
from src.domain.identity import get_main_character_id, get_character_name
from src.config import config
from src.writing.services.versioned_writer import VersionedWriter
from src.agents.drama_planner import DramaPlannerAgent
from src.orchestrator.state import AgentState
from src.writing.loop_store import LoopStore
from src.db.pool import init_writing_progress
from src.writing.controlled_writer import ControlledWriter
from src.writing.planning_contract import PlanningContract
from src.narrative.adaptive import create_adaptive_resolver_with_rollout
from src.narrative.intent import IntentResolver
from src.writing.bootstrap.composition_root import WriterRuntime
from src.writing.controlled_writer import ControlledWriter
from src.writing.projection_service import NarrativeProjectionService
from src.writing.projection_updater import ProjectionUpdater
from src.writing.narrative_intent import NarrativeIntent
from src.writing.planner_output import PlannerOutput
# ========== B2-1A: Contract Sanity Guard ==========
from src.writing.contract_sanity import ContractSanityGuard
from src.writing.world_state import WorldState
from src.writing.realm_authority import DefaultRealmAuthority
# =================================================
from src.writing.validation_v2.audit_writer import AuditContext
from src.writing.validation_v2.bridge_audit_writer import BridgeAuditContext


# 全局 logger
logger = setup_logging("orchestrator.nodes")

# ============================================================
# Phase 15.5: writer_events 提取
# ============================================================
def _extract_writer_events_from_artifact(
    writer_artifact: Optional[Dict[str, Any]]
) -> Optional[List[Dict[str, Any]]]:
    """
    从 writer_artifact 中提取 events，支持三态返回。

    - None  → 数据不可用
    - []    → Writer 明确返回了空 events
    - [...] → 正常可诊断数据
    """
    if not isinstance(writer_artifact, dict):
        return None

    events = writer_artifact.get("events")
    if events is None:
        return None
    if isinstance(events, list):
        return events
    else:
        logger.warning(
            "[Shadow] Invalid writer_artifact.events type: %s",
            type(events).__name__,
        )
        return None
# ============================================================

async def _run_shadow_rewrite(
    scene_id: str,
    original_text: str,
    contract: dict,
    original_validation_result: dict,
    shadow_runner,
    shadow_recorder,
    experiment_id: str = "phase15.3.v1",
    novel_id: str = "",
    volume: int = 1,
    chapter: int = 1,
    scene_idx: int = 0,
    writer_artifact: Optional[Dict[str, Any]] = None,
) -> None:
    """
    异步执行 Shadow Rewrite。

    原则：
    - 任何失败均不得影响生产 Runtime
    - 只记录日志，不抛出异常
    - 不修改 state
    - 不阻塞生产路径
    """
    try:
        # ========== Phase 15.5: 提取 writer_events ==========
        writer_events = _extract_writer_events_from_artifact(writer_artifact)
        # 记录三态信息便于审计
        if writer_events is None:
            logger.debug("[Shadow] writer_events: DATA_UNAVAILABLE")
        elif writer_events == []:
            logger.debug("[Shadow] writer_events: [] (Writer returned empty events)")
        else:
            logger.debug("[Shadow] writer_events: list of %d events", len(writer_events))
        # =====================================================

        logger.info("[Shadow] === calling shadow_runner.submit ===")
        shadow_result = await shadow_runner.submit(
            scene_id=scene_id,
            original_text=original_text,
            contract=contract,
            original_validation_result=original_validation_result,
            experiment_id=experiment_id,
            prompt_version="phase15.4c.contract_reinforced.v1",
            writer_events=writer_events,  # Phase 15.5 新增
        )
        logger.info("[Shadow] === shadow_runner.submit returned, status=%s ===", shadow_result.status.value)
        logger.info("[Shadow] === calling shadow_recorder.record ===")
        await shadow_recorder.record(shadow_result)
        logger.info("[Shadow] === shadow_recorder.record completed ===")
        logger.info(
            "[Shadow] Recorded result: scene=%s status=%s",
            scene_id,
            shadow_result.status.value,
        )

        # ========== 将重写文本写入独立目录（Phase 15.3 Shadow Corpus） ==========
        if shadow_result.rewritten_text and novel_id:
            try:
                shadow_dir = Path(f"data/novels/{novel_id}/shadow/vol_{volume:03d}")
                shadow_dir.mkdir(parents=True, exist_ok=True)
                
                chapter_file = shadow_dir / f"chap_{chapter:03d}.txt"
                
                # 构建场景分隔标记
                scene_marker = f"\n\n<!-- scene {scene_idx:02d} -->\n\n"
                
                # 如果是第一个场景，直接写入；否则追加
                if not chapter_file.exists():
                    chapter_file.write_text(shadow_result.rewritten_text, encoding="utf-8")
                else:
                    with open(chapter_file, "a", encoding="utf-8") as f:
                        f.write(scene_marker)
                        f.write(shadow_result.rewritten_text)
                
                logger.info(f"[Shadow] ✅ Rewritten text appended to {chapter_file} (scene {scene_idx})")
            except Exception as e:
                import traceback
                logger.error(f"[Shadow] ❌ Failed to save rewritten text: {e}\n{traceback.format_exc()}")
        else:
            logger.warning(f"[Shadow] ⚠️ Skip file save: rewritten_text={bool(shadow_result.rewritten_text)}, novel_id={novel_id}")
        # ======================================================================      

    except Exception as e:
        logger.error("[Shadow] Failed for scene=%s: %s", scene_id, e, exc_info=True)

_memory_agent = MemoryAgent()


# ============================================================================
# 辅助函数（保留必要的）
# ============================================================================

async def _load_scene_plans_from_db(
    pool, novel_id: str, volume_num: int, chapter_num: int
) -> Tuple[List[Dict[str, Any]], int]:
    """从 scene_execution_units 表加载指定章节的场景计划列表"""
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
            scene_plans.append(plan)
        return scene_plans, len(scene_plans)


def get_memory_agent() -> MemoryAgent:
    """获取全局 MemoryAgent 实例"""
    return _memory_agent


def _keyword_analyze(user_input: str) -> tuple[str, list[str]]:
    """简单的意图识别（用于 analyze_node）"""
    lower = user_input.lower()
    if any(kw in lower for kw in ["write", "code", "implement", "function", "class", "create"]):
        intent = "code_generation"
    elif any(kw in lower for kw in ["explain", "what is", "how does", "tell me", "research", "knowledge"]):
        intent = "research"
    else:
        intent = "general_chat"
    return intent, [user_input]


def _is_complex_task(user_input: str) -> bool:
    """判断任务是否复杂（长度 > 200 字符）"""
    return len(user_input) > 200


async def _save_scene_to_file(state: AgentState, raw_text: str) -> None:
    """将场景正文追加到章节文件"""
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


async def _skip_scene(state: AgentState):
    pool = get_db_pool()
    if not pool:
        return

    # 计算 next state（统一语义）
    new_scene_idx = (state.current_scene_index or 0) + 1
    total_scenes = state.total_scenes_in_chapter or 0
    chapter_finished_for_db = (total_scenes > 0 and new_scene_idx >= total_scenes)

    if chapter_finished_for_db:
        next_volume = state.current_volume
        next_chapter = state.current_chapter + 1
        next_scene = 0
        chapter_completed_flag = True
    else:
        next_volume = state.current_volume
        next_chapter = state.current_chapter
        next_scene = new_scene_idx
        chapter_completed_flag = False

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO writing_progress
                    (project_id, current_volume, current_chapter, current_scene, chapter_completed, last_updated)
                VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (project_id) DO UPDATE SET
                    current_volume = EXCLUDED.current_volume,
                    current_chapter = EXCLUDED.current_chapter,
                    current_scene = EXCLUDED.current_scene,
                    chapter_completed = EXCLUDED.chapter_completed,
                    last_updated = NOW()
                """,
                state.novel_id, next_volume, next_chapter, next_scene, chapter_completed_flag
            )
            await conn.execute(
                """
                UPDATE scene_execution_units
                SET status = 'skipped', completed_at = NOW(), updated_at = NOW()
                WHERE novel_id = $1 AND volume_num = $2 AND chapter_num = $3 AND scene_index = $4
                """,
                state.novel_id, state.current_volume, state.current_chapter, state.current_scene_index
            )

# ============================================================================
# LangGraph 节点函数
# ============================================================================

async def load_memory_node(state: AgentState) -> dict[str, Any]:
    """加载记忆上下文"""
    return await _memory_agent.run(state)


async def save_memory_node(state: AgentState) -> dict[str, Any]:
    """保存小说大纲到数据库，并同步 writing_progress，初始化 character_arcs"""
    logger.info(f"save_memory_node: state.outline={state.outline}, state.task_type={state.task_type}")
    logger.info(f"save_memory_node: outline type={type(state.outline)}, value={state.outline}")
    logger.info(f"Save memory for {state.novel_id}, outline exists: {state.outline is not None}")
    logger.info(f"outline value: {state.outline}")  # 新增调试
    
    logger.info(f"=== save_memory_node called ===")
    logger.info(f"novel_id={state.novel_id}, outline is None? {state.outline is None}")
    if state.outline:
        logger.info(f"outline keys: {list(state.outline.keys())}")
        logger.info(f"volumes count: {len(state.outline.get('volumes', []))}")
    else:
        logger.warning("state.outline is None, cannot save")    
    
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
            await init_writing_progress(
                state.novel_id,
                volume=state.current_volume,
                chapter=state.current_chapter,
                scene=state.current_scene_index if state.current_scene_index is not None else 0,
                chapter_completed=False
            )
            logger.info(f"Synced writing_progress for {state.novel_id} (volume={state.current_volume}, chapter={state.current_chapter}, scene={state.current_scene_index})")

            # ========== 初始化 character_arcs（如果未初始化） ==========
            if state.compressed_state:
                if isinstance(state.compressed_state, dict):
                    compressed = state.compressed_state
                else:
                    compressed = state.compressed_state.model_dump() if hasattr(state.compressed_state, 'model_dump') else {}

                existing_arcs = compressed.get("character_arcs", {})
                if not existing_arcs and state.outline:
                    arcs = {}
                    volumes = state.outline.get("volumes", [])
                    # 使用主角 ID 作为弧线标识的一部分
                    protagonist_id = get_main_character_id()
                    for vol in volumes:
                        vol_num = vol.get("volume_num")
                        core_conflict = vol.get("core_conflict", "")
                        if core_conflict:
                            arcs[f"volume_{vol_num}_conflict_{protagonist_id}"] = "open"
                    if arcs:
                        compressed["character_arcs"] = arcs
                        if isinstance(state.compressed_state, dict):
                            state.compressed_state = compressed
                        else:
                            state.compressed_state.character_arcs = arcs
                        logger.info(f"Initialized character_arcs for novel {state.novel_id}: {len(arcs)} arcs")
                    else:
                        logger.debug("No core conflicts found in outline to initialize arcs")
            # ========================================================       
    else:
        logger.warning(f"state.outline is empty for {state.novel_id}, cannot save")

    # 在最后添加审计
    await audit_state(state, "save_memory")
    return {"metadata": state.metadata, "novel_id": state.novel_id}


async def analyze_node(state: AgentState) -> dict[str, Any]:
    """分析用户意图（仅用于非小说任务）"""
    intent, subtasks = _keyword_analyze(state.user_input)
    return {
        "intent": intent,
        "subtasks": subtasks,
        "is_complex": _is_complex_task(state.user_input),
        "current_node": "analyze"
    }


async def plan_node(state: AgentState) -> dict:
    logger.error("🚨🚨 PLAN_NODE_V2 IS EXECUTING 🚨🚨")  # 强制醒目日志
    logger.info(f"plan_node called with chapter={state.current_chapter}, task_type={state.task_type}")

    # --- 1. 小说大纲生成 ---
    if state.task_type == "novel_outline":
        planner = PlannerAgent()
        result = await planner.run(state)
        outline = result.get("outline")
        if not outline:
            logger.error("Failed to generate outline")
            return {"error": "Outline generation failed"}
        return {"outline": outline}

    # --- 2. 场景计划生成（核心修改）---
    if state.task_type == "scene_plan":
        # 配置 IntentResolver
        if config.adaptive_runtime_enabled:
            conflict_resolver = create_adaptive_resolver_with_rollout(
                rollout_percentage=config.adaptive_rollout_percentage,
                enable_telemetry=True,
                novel_id=state.novel_id,
                chapter=state.current_chapter,
                scene=state.current_scene_index,
            )
            resolver = IntentResolver(conflict_resolver=conflict_resolver)
            logger.info(f"Adaptive runtime enabled, rollout={config.adaptive_rollout_percentage}%")
        else:
            resolver = IntentResolver()
            logger.info("Adaptive runtime disabled, using rule selector")

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
            metadata=state.metadata,
            intent_resolver=resolver,
        )

        result = await ScenePlanningService.execute(cmd)

        if result.error:
            return StatePatch(error=result.error).to_dict()

        patch = result.state_patch or StatePatch()

        # 无条件设置 planner_outputs，即使为空列表
        patch.planner_outputs = result.planner_outputs

        if result.planner_outputs:
            logger.info(f"✅ plan_node: planner_outputs propagated count={len(result.planner_outputs)}")
            first = result.planner_outputs[0]
            intent_data = first.get("narrative_intent")
            if intent_data:
                from src.writing.narrative_intent import NarrativeIntent
                patch.narrative_intent = NarrativeIntent.from_dict(intent_data) if isinstance(intent_data, dict) else intent_data
                logger.info(f"✅ plan_node: set narrative_intent from first scene")
        else:
            logger.info("plan_node: planner_outputs is empty, setting empty list")

        payload = patch.to_dict()
        logger.info(f"plan_node payload audit: planner_outputs count={len(payload.get('planner_outputs', []))}, has_narrative_intent={'narrative_intent' in payload}")
        
        # ========== P0 诊断：plan_node 返回载荷 ==========
        logger.critical(
            "PLAN_NODE_RETURN_PAYLOAD: planner_outputs_count=%d ids=%s keys=%s",
            len(payload.get("planner_outputs", [])),
            [
                p.get("narrative_intent", {}).get("intent_id")
                if isinstance(p, dict)
                else type(p).__name__
                for p in payload.get("planner_outputs", [])
            ],
            list(payload.keys())
        )
        # =================================================
        
        return payload
        # ================================================================

    # --- 3. 其他任务（回退）---
    planner = PlannerAgent()
    result = await planner.run(state)
    return result


# ============================================================================
# writer_node 完整函数（已修改，增加强制传递）
# ============================================================================
async def writer_node(state: AgentState, runtime: WriterRuntime) -> dict:
    # ========== P0 诊断：writer_node 输入状态 ==========
    logger.critical(
        "WRITER_NODE_ENTRY_STATE: planner_outputs_count=%d scene_plan_list_len=%d",
        len(state.planner_outputs or []),
        len(state.scene_plan_list or [])
    )
    if state.planner_outputs:
        logger.critical(
            "WRITER_NODE_ENTRY_INTENTS: %s",
            [
                p.get("narrative_intent", {}).get("intent_id")
                for p in state.planner_outputs
                if isinstance(p, dict)
            ]
        )
    # =================================================

    # ========== 修复：从 metadata 恢复 planner_outputs ==========
    if not state.planner_outputs and state.metadata.get("planner_outputs"):
        state.planner_outputs = state.metadata["planner_outputs"]
        logger.info(f"writer_node: restored planner_outputs from metadata (count={len(state.planner_outputs)})")
    # ================================================================

    logger.info("WritingAgent starting")

    """
    Writer 节点 - 使用 ControlledWriter 作为默认执行引擎。

    Args:
        state: AgentState，包含场景计划、世界状态等
        runtime: WriterRuntime，由 Composition Root 注入

    Returns:
        dict: StatePatch 更新
    """

    # ========== PHASE 15.0 AUDIT ==========
    logger.critical(
        "[PHASE15] writer_node_input planner_outputs_count=%s scene_plan_list_len=%s scene_text_len=%s",
        len(state.planner_outputs) if state.planner_outputs else 0,
        len(state.scene_plan_list) if state.scene_plan_list else 0,
        len(state.scene_text) if state.scene_text else 0
    )
    if state.planner_outputs:
        first = state.planner_outputs[0]
        logger.critical(
            "[PHASE15] writer_node_input first_planner_output keys=%s has_intent=%s",
            list(first.keys()) if isinstance(first, dict) else "not_dict",
            "narrative_intent" in (first if isinstance(first, dict) else {})
        )
    # =====================================

    # ========== 0. 提前获取 Planning Contract ==========
    planning_contract = getattr(state, 'planning_contract', None)

    # ========== 1. 从数据库加载当前激活的 Loop ==========
    if state.novel_id:
        pool = get_db_pool()
        if pool:
            loop_store = LoopStore(pool)
            active_loop = await loop_store.get_active_loop(state.novel_id)
            if active_loop:
                state.metadata["active_loop"] = {
                    "id": str(active_loop.id),
                    "title": active_loop.title,
                    "description": active_loop.description,
                    "progress": active_loop.progress,
                }
                logger.info(f"✅ Loaded active_loop from DB: {active_loop.title}")

    # ========== 2. 实验组配置 ==========
    state.metadata["experiment_group"] = "loop"

    # ========== 3. 获取当前场景计划（防御性） ==========
    scene_plan_list = state.scene_plan_list
    current_idx = state.current_scene_index if state.current_scene_index is not None else 0

    # ===== 新的强类型读取逻辑 =====
    scene_plan_list = state.scene_plan_list
    if scene_plan_list is None:
        logger.error("writer_node: scene_plan_list is None (missing state)")
        return StatePatch(error="scene_plan_list missing from state").to_dict()
    if not scene_plan_list:
        logger.error("writer_node: scene_plan_list is empty")
        return StatePatch(error="scene_plan_list empty").to_dict()

    if current_idx >= len(scene_plan_list):
        logger.error(f"writer_node: invalid scene index {current_idx} (len={len(scene_plan_list)})")
        return StatePatch(error="Invalid scene index").to_dict()

    current_scene_plan = scene_plan_list[current_idx]
    state.scene_plan = current_scene_plan
    state.metadata["current_scene_plan"] = current_scene_plan

    # ========== PHASE 15.0 AUDIT ==========
    logger.critical(
        "[PHASE15] writer_node_current_scene_plan scene_id=%s characters=%s must_events_count=%s",
        current_scene_plan.get("scene_id", "unknown"),
        current_scene_plan.get("characters", []),
        len(current_scene_plan.get("must_events", []))
    )
    # =====================================

    # ========== 4. 提取 Planning Contract ==========
    planning_contract = current_scene_plan.get("planning_contract")
    if planning_contract:
        state.planning_contract = planning_contract
        scene_id = planning_contract.get('scene_id', 'unknown')
        logger.info(f"✅ 从 scene_plan 提取 Planning Contract: {scene_id}")
    else:
        logger.warning(f"⚠️ scene_plan 中无 planning_contract (scene_idx={current_idx})")

    # ========== B2-1A: Contract Sanity Guard (阻断模式) ==========
    if planning_contract is not None:
        try:
            guard = ContractSanityGuard(
                realm_authority=DefaultRealmAuthority()
                #realm_authority=None  # B2-1A: 暂不注入 RealmAuthority
            )
            
            # 构建 WorldState
            world_state = WorldState.from_dict(state.current_state) if state.current_state else WorldState()
            
            # scene_id 从 planning_contract 自身获取（统一来源）
            scene_id = (
                planning_contract.get("scene_id", "unknown")
                if isinstance(planning_contract, dict)
                else getattr(planning_contract, "scene_id", "unknown")
            )
            
            # contract_id: 不伪造，只有真实存在时才传入
            contract_id = (
                planning_contract.get("contract_id", "")
                if isinstance(planning_contract, dict)
                else getattr(planning_contract, "contract_id", "")
            )
            
            sanity_result = guard.check(
                contract=planning_contract,
                world_state=world_state,
                scene_id=scene_id,
                contract_id=contract_id or "",
            )
            
            if not sanity_result.valid:
                logger.warning(
                    f"[writer_node] Contract sanity check failed: {sanity_result.summary}"
                )
                # ✅ 阻断：非法 Contract 不进入 Writer
                return StatePatch(
                    error=f"Contract invalid: {sanity_result.violations[0].description}",
                    metadata={
                        "sanity_result": sanity_result.to_dict(),
                        "sanity_check_id": sanity_result.check_id,
                    },
                    phase=WorkflowPhase.VALIDATING,
                ).to_dict()
                
        except Exception as e:
            logger.error(f"[writer_node] ContractSanityGuard critical error: {e}", exc_info=True)
            
            # 提取审计上下文（尽力而为）
            audit_context = {
                "scene_id": scene_id,
                "contract_id": contract_id or "",
                "error": str(e),
                "error_type": type(e).__name__,
            }
            
            # ✅ 阻断：Guard 内部异常同样阻止 Writer 执行（fail-closed）
            return StatePatch(
                error=f"Contract Sanity Guard internal error: {e}",
                metadata={
                    "sanity_error": str(e),
                    "sanity_error_type": type(e).__name__,
                    "sanity_audit_context": audit_context,
                    "sanity_blocked": True,
                    "sanity_block_reason": "GUARD_INTERNAL_ERROR",
                },
                phase=WorkflowPhase.VALIDATING,
            ).to_dict()
    # ================================================================

    # ========== P0 诊断：确认 state.planning_contract 赋值成功 ==========
    logger.critical(
        "WRITER_NODE_SET_PLANNING_CONTRACT: scene_id=%s, type=%s, value_type=%s",
        planning_contract.get('scene_id', 'unknown') if planning_contract else 'None',
        type(planning_contract).__name__ if planning_contract else 'None',
        type(state.planning_contract).__name__ if state.planning_contract else 'None'
    )
    # ====================================================================

    # ========== 5. 处理 Drama Structure ==========
    if state.drama_structure is None:
        drama_from_plan = current_scene_plan.get("drama")
        if drama_from_plan:
            state.drama_structure = drama_from_plan
            if "scene_role" in drama_from_plan:
                current_scene_plan["scene_role"] = drama_from_plan["scene_role"]
                logger.info(f"[writer_node] Set scene_role from drama: {drama_from_plan['scene_role']}")
        else:
            logger.info(f"[writer_node] No drama_structure in scene plan, generating on the fly")
            temp_drama_state = AgentState(
                scene_plan=current_scene_plan,
                novel_id=state.novel_id,
                current_volume=state.current_volume,
                current_chapter=state.current_chapter,
                current_state=state.current_state,
            )
            try:
                drama_planner = DramaPlannerAgent()
                result = await drama_planner.run(temp_drama_state)
                drama_struct = result.get("drama_structure", {})
                if drama_struct:
                    state.drama_structure = drama_struct
                    if "scene_role" in drama_struct:
                        current_scene_plan["scene_role"] = drama_struct["scene_role"]
                    await _update_scene_plan_drama(
                        state.novel_id, state.current_volume, state.current_chapter,
                        current_idx, drama_struct
                    )
                    logger.info(f"[writer_node] Generated and saved drama_structure")
                else:
                    logger.warning(f"[writer_node] Failed to generate drama_structure")
            except Exception as e:
                logger.error(f"[writer_node] Drama generation failed: {e}")

    # ========== 6. 从强类型或 metadata 恢复 planner_outputs ==========
    planner_outputs = state.planner_outputs
    if planner_outputs is None:
        logger.error("writer_node: planner_outputs is None (missing state)")
        return StatePatch(error="planner_outputs missing from state").to_dict()

    # ========== P0 修复：如果 planner_outputs 为空但 scene_plan_list 有值，重建 ==========
    if not planner_outputs and scene_plan_list:
        logger.warning("writer_node: planner_outputs empty but scene_plan_list not empty, attempting rebuild")
        try:
            from src.writing.planner_output import PlannerOutput
            from src.writing.narrative_intent import NarrativeIntent, SceneRole
            from src.writing.planning_contract import PlanningContract

            rebuilt = []
            for idx, scene in enumerate(scene_plan_list):
                intent_data = scene.get("narrative_intent") or {}
                scene_role_str = intent_data.get("scene_role", "transition")
                try:
                    scene_role = SceneRole(scene_role_str)
                except ValueError:
                    scene_role = SceneRole.TRANSITION

                narrative_intent = NarrativeIntent(
                    intent_id=intent_data.get("intent_id", f"rebuilt_{idx}"),
                    scene_role=scene_role,
                    objective=intent_data.get("objective", scene.get("goal", "推进剧情")),
                    preconditions=[],
                    beats=[],
                    consequences=[],
                    interaction_plan=None,
                )

                planning_contract_data = scene.get("planning_contract")
                if planning_contract_data:
                    if isinstance(planning_contract_data, dict):
                        contract = PlanningContract(**planning_contract_data)
                    else:
                        contract = planning_contract_data
                    rebuilt.append({
                        "narrative_intent": narrative_intent.to_dict(),
                        "execution_contract": contract.to_dict() if hasattr(contract, 'to_dict') else contract,
                    })
                else:
                    logger.warning(f"Scene {idx} missing planning_contract, using empty contract")
                    from src.writing.planning_contract import PlanningContract, Intent, Execution, ContractMetadata
                    contract = PlanningContract(
                        scene_id=scene.get("scene_id", f"rebuilt_{idx}"),
                        intent=Intent(
                            goal=scene.get("goal", ""),
                            conflict=scene.get("conflict", ""),
                            expected_outcome=scene.get("outcome", ""),
                        ),
                        execution=Execution(),
                        observables=Observables(),
                        metadata=ContractMetadata(
                            chapter=state.current_chapter,
                            scene_index=idx,
                        ),
                    )
                    rebuilt.append({
                        "narrative_intent": narrative_intent.to_dict(),
                        "execution_contract": contract.to_dict(),
                    })

            if rebuilt:
                planner_outputs = rebuilt
                state.planner_outputs = rebuilt
                logger.info(f"writer_node: rebuilt planner_outputs from scene_plan_list (count={len(rebuilt)})")
            else:
                logger.error("writer_node: failed to rebuild planner_outputs from scene_plan_list")
                return StatePatch(error="Failed to rebuild planner_outputs").to_dict()
        except Exception as e:
            logger.error(f"writer_node: failed to rebuild planner_outputs: {e}", exc_info=True)
            return StatePatch(error=f"Failed to rebuild planner_outputs: {e}").to_dict()
    # ====================================================================

    if not planner_outputs:
        logger.warning("writer_node: planner_outputs is empty, continuing with no scenes")
        if scene_plan_list:
            logger.error("writer_node: planner_outputs empty but scene_plan_list not empty")
            return StatePatch(error="planner_outputs empty but scene_plan_list not empty").to_dict()

    logger.info(f"writer_node: planner_outputs count={len(planner_outputs)}")

    narrative_intent = None
    current_idx = state.current_scene_index if state.current_scene_index is not None else 0

    if planner_outputs and current_idx < len(planner_outputs):
        planner_output = planner_outputs[current_idx]
        intent_data = planner_output.get("narrative_intent")
        if intent_data:
            from src.writing.narrative_intent import NarrativeIntent
            if isinstance(intent_data, dict):
                narrative_intent = NarrativeIntent.from_dict(intent_data)
            else:
                narrative_intent = intent_data
            logger.info(
                f"✅ writer_node: 解析 narrative_intent "
                f"(intent_id={narrative_intent.intent_id})"
            )
            logger.critical(
                "[PHASE15] writer_node_narrative_intent intent_id=%s scene_role=%s objective=%s",
                narrative_intent.intent_id,
                narrative_intent.scene_role.value if hasattr(narrative_intent.scene_role, 'value') else str(narrative_intent.scene_role),
                narrative_intent.objective[:50]
            )
        else:
            logger.warning(f"writer_node: planner_outputs[{current_idx}] 缺少 narrative_intent")
    else:
        logger.warning(
            f"writer_node: planner_outputs empty or index out of range "
            f"(len={len(planner_outputs)}, idx={current_idx})"
        )

    logger.info(f"writer_node input planner_outputs count={len(planner_outputs)}")
    # ================================================================
    # ========== 15.7-B1 Fix: 确保 narrative_intent 从 planner_outputs 绑定到 state ==========
    if planner_outputs and current_idx < len(planner_outputs):
        current_output = planner_outputs[current_idx]
        if isinstance(current_output, dict):
            intent_data = current_output.get("narrative_intent")
            if intent_data:
                if isinstance(intent_data, dict):
                    state.narrative_intent = NarrativeIntent.from_dict(intent_data)
                else:
                    state.narrative_intent = intent_data
                logger.critical(
                    "[15.7-B1] narrative_intent bound: intent_id=%s, scene_role=%s",
                    state.narrative_intent.intent_id,
                    state.narrative_intent.scene_role.value if hasattr(state.narrative_intent.scene_role, 'value') else str(state.narrative_intent.scene_role)
                )
            else:
                logger.critical("[15.7-B1] narrative_intent missing in planner_outputs[%d]", current_idx)
        elif hasattr(current_output, 'narrative_intent'):
            state.narrative_intent = current_output.narrative_intent
            logger.critical(
                "[15.7-B1] narrative_intent bound (object): intent_id=%s",
                state.narrative_intent.intent_id
            )
        else:
            logger.critical("[15.7-B1] planner_outputs[%d] has no narrative_intent field", current_idx)
    else:
        logger.warning("[15.7-B1] planner_outputs empty or index out of range, narrative_intent not bound")
    # ===================================================================================
    # ========== 7. Phase 13.2.1: 构建 WritingContract ==========
    from src.writing.contracts import WritingContract, WritingConstraints, WritingGoal
    from src.writing.scene_execution_context import SceneExecutionContext

    if not planning_contract:
        planning_contract = state.metadata.get("planning_contract")

    scene_context = SceneExecutionContext(
        chapter_id=f"{state.novel_id}_c{state.current_chapter}",
        scene_id=current_scene_plan.get("scene_id", f"scene_{state.current_chapter}_{state.current_scene_index}"),
        scene_role=current_scene_plan.get("scene_role", "transition"),
        dramatic_function=current_scene_plan.get("dramatic_function", "transition"),
        characters=current_scene_plan.get("characters", []),
        location=current_scene_plan.get("location", "未知"),
        time=current_scene_plan.get("time", "未知"),
    )

    constraints = WritingConstraints(
        must_events=current_scene_plan.get("must_events", []),
        forbidden_events=current_scene_plan.get("forbidden_events", []),
    )

    narrative_intent = state.narrative_intent

    writing_goal = None
    if planning_contract and isinstance(planning_contract, dict):
        intent = planning_contract.get("intent", {})
        if intent:
            writing_goal = WritingGoal(
                goal=intent.get("goal", ""),
                conflict=intent.get("conflict", ""),
                expected_outcome=intent.get("expected_outcome", ""),
            )
    elif current_scene_plan:
        writing_goal = WritingGoal(
            goal=current_scene_plan.get("goal", ""),
            conflict=current_scene_plan.get("conflict", ""),
            expected_outcome=current_scene_plan.get("outcome", ""),
        )

    planning_contract_obj = None
    if planning_contract:
        if isinstance(planning_contract, dict):
            from src.writing.planning_contract import PlanningContract
            try:
                planning_contract_obj = PlanningContract(**planning_contract)
            except Exception as e:
                logger.warning(f"Failed to convert planning_contract to object: {e}")
                planning_contract_obj = planning_contract
        else:
            planning_contract_obj = planning_contract

    writing_contract = WritingContract(
        scene_context=scene_context,
        narrative_intent=state.narrative_intent,   # ← 使用 state 中的绑定值
        constraints=constraints,
        writing_goal=writing_goal,
        execution_contract=planning_contract_obj,
    )
    logger.critical(
        "[15.7-B1] WritingContract binding: "
        "intent=%s, execution_contract=%s, units=%d",
        getattr(state.narrative_intent, "intent_id", None),
        planning_contract_obj is not None,
        len(planning_contract_obj.execution.units)
        if planning_contract_obj 
        and hasattr(planning_contract_obj, 'execution') 
        and hasattr(planning_contract_obj.execution, 'units')
        else 0,
    )
    logger.critical(
        "[PHASE15] writer_node_writing_contract scene_id=%s characters=%s has_intent=%s",
        writing_contract.scene_context.scene_id,
        writing_contract.scene_context.characters,
        writing_contract.narrative_intent is not None
    )

    logger.critical(
        "[15.7-B1] rewriter being passed to ControlledWriter: type=%s, is None? %s",
        type(runtime.runtime_services.rewriter).__name__ if runtime.runtime_services.rewriter is not None else "None",
        runtime.runtime_services.rewriter is None
    )

    cw = ControlledWriter(
        runtime_services=runtime.runtime_services,
        rewriter=runtime.runtime_services.rewriter,
    )

    # ========== 8. 辅助函数：构建包含关键状态的 StatePatch ==========
    def _build_patch(
        scene_text: str,
        final_answer: str,
        metadata: Dict[str, Any] = None,
        error: str = None,
        phase: WorkflowPhase = None,
        writer_artifact: Optional[Dict[str, Any]] = None,
    ) -> StatePatch:
        patch_metadata = metadata or {}
        if planner_outputs:
            patch_metadata["planner_outputs"] = planner_outputs
        if narrative_intent:
            patch_metadata["narrative_intent"] = narrative_intent.to_dict() if hasattr(narrative_intent, 'to_dict') else narrative_intent
        if current_scene_plan:
            patch_metadata["current_scene_plan"] = current_scene_plan

        patch = StatePatch(
            scene_text=scene_text,
            final_answer=final_answer,
            planner_outputs=planner_outputs,
            narrative_intent=narrative_intent,
            scene_plan=current_scene_plan,
            metadata=patch_metadata,
            writer_artifact=writer_artifact,
        )
        if error:
            patch.error = error
        if phase:
            patch.phase = phase
        return patch

    # ========== 9. 执行写入 ==========
    patch = None

    if getattr(config, 'controlled_writer_enabled', True):
        # ========== Phase 15.7-B1: 健壮提取 exec_units ==========
        exec_units = []
        if planning_contract:
            if isinstance(planning_contract, dict):
                exec_units = planning_contract.get("execution", {}).get("units", [])
            elif hasattr(planning_contract, 'execution'):
                # PlanningContract 对象
                if hasattr(planning_contract.execution, 'units'):
                    exec_units = planning_contract.execution.units
                elif isinstance(planning_contract.execution, dict):
                    exec_units = planning_contract.execution.get("units", [])
        logger.critical(
            "[15.7-B1] exec_units extraction: count=%d, type=%s",
            len(exec_units),
            type(planning_contract).__name__ if planning_contract else "None"
        )
        # =====================================================
        if len(exec_units) >= 1:
            logger.info(f"🚀 使用 ControlledWriter: {len(exec_units)} 个执行单元")
            try:
                logger.critical("[15.7-B1] BEFORE execute: cw = %s", type(cw).__name__)
                result = await cw.execute(writing_contract)
                logger.critical("[15.7-B1] AFTER execute: result = %s", result)
                # ========== Phase 15.7-A 临时验证 ==========
                logger.critical(
                    "[15.7-A] ControlledWriteResult: "
                    "original_text_len=%d, "
                    "rewritten_text_is_none=%s, "
                    "rewrite_attempted=%s",
                    len(result.original_text) if result.original_text else 0,
                    result.rewritten_text is None,
                    result.rewrite_attempted,
                )
                # =============================================

                if result.text:
                    logger.critical(
                        "[PHASE15] writer_node_controlled_writer_result text_len=%s events_count=%s",
                        len(result.text),
                        len(result.events)
                    )
                    logger.critical(
                        "[PHASE15] writer_node_controlled_writer_result contains_linyi=%s contains_protagonist=%s abcd=%s",
                        "林逸" in result.text,
                        "protagonist" in result.text,
                        re.findall(r'\b[A-D]\b', result.text)
                    )

                    patch_metadata = {
                        "controlled_writer": {
                            "segments": result.segments_used,
                            "succeeded": result.segments_succeeded,
                            "fallback": result.fallback_used,
                            "time": result.execution_time,
                        }
                    }
                    if state.metadata.get("active_loop"):
                        patch_metadata["active_loop"] = state.metadata["active_loop"]
                    if planning_contract:
                        patch_metadata["planning_contract"] = planning_contract
                    patch_metadata["current_scene_plan"] = current_scene_plan

                    # ========== Phase 15.7-B1: 构造 writer_artifact v1.1 ==========
                    # ========== Phase 15.8 Commit 1: writer_artifact v1.2 ==========
                    selection_dict = (
                        result.selection.to_dict() if result.selection else None
                    )
                    writer_artifact = {
                        "schema_version": "1.2",
                        "scene_text": result.text,
                        "events": result.events,
                        "foreshadowing": [],
                        "original_text": result.original_text,
                        "rewritten_text": result.rewritten_text,
                        "rewrite_attempted": result.rewrite_attempted,
                        "rewrite_failure_reason": result.rewrite_failure_reason,
                        # Phase 15.8 Commit 1
                        "selection": selection_dict,
                        "final_text": result.original_text,
                        "selected_source": (
                            selection_dict.get("selected_source")
                            if selection_dict else "original"
                        ),
                        "selection_reason": (
                            selection_dict.get("selection_reason")
                            if selection_dict else "unknown"
                        ),
                        "validation_original": None,
                        "validation_rewritten": None,
                    }
                    logger.critical(
                        "WRITER_NODE_ARTIFACT: schema_version=1.2, events_len=%d, text_len=%d, "
                        "rewrite_attempted=%s, selected_source=%s, selection_reason=%s",
                        len(result.events),
                        len(result.text),
                        result.rewrite_attempted,
                        writer_artifact["selected_source"],
                        writer_artifact["selection_reason"],
                    )
                    # =====================================================================

                    patch = _build_patch(
                        scene_text=result.text,
                        final_answer=result.text,
                        writer_artifact=writer_artifact,
                        metadata=patch_metadata,
                    )
                else:
                    logger.error("❌ ControlledWriter 返回空文本")
                    # ========== Phase 15.7-B1: 构造空 writer_artifact ==========
                    # ========== Phase 15.8 Commit 1: writer_artifact v1.2 (空结果) ==========
                    writer_artifact = {
                        "schema_version": "1.2",
                        "scene_text": "",
                        "events": [],
                        "foreshadowing": [],
                        "original_text": result.original_text if result.original_text else "",
                        "rewritten_text": result.rewritten_text,
                        "rewrite_attempted": result.rewrite_attempted,
                        "rewrite_failure_reason": result.rewrite_failure_reason or "ControlledWriter returned empty",
                        # Phase 15.8 Commit 1
                        "selection": None,
                        "final_text": "",
                        "selected_source": "original",
                        "selection_reason": "writer_no_text",
                        "validation_original": None,
                        "validation_rewritten": None,
                    }
                    # ======================================================================
                    patch = _build_patch(
                        scene_text="",
                        final_answer="",
                        writer_artifact=writer_artifact,
                        metadata={"error": "ControlledWriter 返回空结果"},
                        error="ControlledWriter 返回空结果",
                        phase=WorkflowPhase.VALIDATING,
                    )
            except Exception as e:
                logger.exception(f"❌ ControlledWriter 执行失败: {e}")
                # ========== Phase 15.7-B1: 构造空 writer_artifact ==========
                # ========== Phase 15.8 Commit 1: writer_artifact v1.2 (异常) ==========
                writer_artifact = {
                    "schema_version": "1.2",
                    "scene_text": "",
                    "events": [],
                    "foreshadowing": [],
                    "original_text": "",
                    "rewritten_text": None,
                    "rewrite_attempted": False,
                    "rewrite_failure_reason": f"ControlledWriter exception: {e}",
                    # Phase 15.8 Commit 1
                    "selection": None,
                    "final_text": "",
                    "selected_source": "original",
                    "selection_reason": "writer_exception",
                    "validation_original": None,
                    "validation_rewritten": None,
                }
                # ======================================================================
                patch = _build_patch(
                    scene_text="",
                    final_answer="",
                    writer_artifact=writer_artifact,
                    metadata={"error": f"ControlledWriter 失败: {e}"},
                    error=f"ControlledWriter 失败: {e}",
                    phase=WorkflowPhase.VALIDATING,
                )
        else:
            logger.info(f"📝 使用单次写入（单元数 {len(exec_units)} <= 2）")
            from src.writing.services.writing import WritingService
            from src.writing.services.models import WritingCommand

            cmd = WritingCommand(
                novel_id=state.novel_id,
                volume=state.current_volume,
                chapter=state.current_chapter,
                scene_idx=current_idx,
                scene_plan=current_scene_plan,
                current_state=state.current_state,
                writing_feedback=getattr(state, "writing_feedback", ""),
                narrative_blueprint=state.narrative_blueprint,
                knowledge_deltas=state.knowledge_deltas,
                character_intent=state.character_intent,
                metadata=state.metadata,
                execution_contract=planning_contract,
            )
            result = await WritingService.execute(cmd)

            logger.critical(
                "[PHASE15] writer_node_writing_service_result error=%s text_len=%s events_count=%s",
                result.error,
                len(result.scene_text) if result.scene_text else 0,
                len(result.events) if result.events else 0
            )
            if result.scene_text:
                logger.critical(
                    "[PHASE15] writer_node_writing_service_result contains_linyi=%s abcd=%s",
                    "林逸" in result.scene_text,
                    re.findall(r'\b[A-D]\b', result.scene_text)
                )

            if result.error:
                patch = _build_patch(
                    scene_text=result.scene_text or "",
                    final_answer=result.scene_text or "",
                    metadata={"error": f"单次写入失败: {result.error}"},
                    error=f"单次写入失败: {result.error}",
                    phase=WorkflowPhase.VALIDATING,
                )
            else:
                patch_metadata = {}
                if state.metadata.get("active_loop"):
                    patch_metadata["active_loop"] = state.metadata["active_loop"]
                if planning_contract:
                    patch_metadata["planning_contract"] = planning_contract
                patch_metadata["current_scene_plan"] = current_scene_plan

                # ========== Phase 15.7-B1: 单次写入路径 writer_artifact v1.1 ==========
                # ========== Phase 15.8 Commit 1: 单次写入 writer_artifact v1.2 ==========
                writer_artifact = {
                    "schema_version": "1.2",
                    "scene_text": result.scene_text or "",
                    "events": result.events or [],
                    "foreshadowing": [],
                    "original_text": result.scene_text or "",
                    "rewritten_text": None,
                    "rewrite_attempted": False,
                    "rewrite_failure_reason": "Single-pass writer, no rewrite available",
                    # Phase 15.8 Commit 1
                    "selection": None,
                    "final_text": result.scene_text or "",
                    "selected_source": "original",
                    "selection_reason": "single_pass_writer",
                    "validation_original": None,
                    "validation_rewritten": None,
                }
                logger.critical(
                    "WRITER_NODE_ARTIFACT: schema_version=1.2, events_len=%d, text_len=%d (WritingService path)",
                    len(result.events or []),
                    len(result.scene_text or "")
                )
                # =====================================================================

                patch = _build_patch(
                    scene_text=result.scene_text or "",
                    final_answer=result.scene_text or "",
                    writer_artifact=writer_artifact,
                    metadata=patch_metadata,
                )
    else:
        logger.info("ℹ️ ControlledWriter 被禁用，使用单次写入")
        from src.writing.services.writing import WritingService
        from src.writing.services.models import WritingCommand

        cmd = WritingCommand(
            novel_id=state.novel_id,
            volume=state.current_volume,
            chapter=state.current_chapter,
            scene_idx=current_idx,
            scene_plan=current_scene_plan,
            current_state=state.current_state,
            writing_feedback=getattr(state, "writing_feedback", ""),
            narrative_blueprint=state.narrative_blueprint,
            knowledge_deltas=state.knowledge_deltas,
            character_intent=state.character_intent,
            metadata=state.metadata,
            execution_contract=planning_contract,
        )
        result = await WritingService.execute(cmd)
        if result.error:
            patch = _build_patch(
                scene_text=result.scene_text or "",
                final_answer=result.scene_text or "",
                metadata={"error": f"单次写入失败: {result.error}"},
                error=f"单次写入失败: {result.error}",
                phase=WorkflowPhase.VALIDATING,
            )
        else:
            patch_metadata = {}
            if state.metadata.get("active_loop"):
                patch_metadata["active_loop"] = state.metadata["active_loop"]
            if planning_contract:
                patch_metadata["planning_contract"] = planning_contract
            patch_metadata["current_scene_plan"] = current_scene_plan

            writer_artifact = {
                "schema_version": "1.2",
                "scene_text": result.scene_text or "",
                "events": result.events or [],
                "foreshadowing": [],
                "original_text": result.scene_text or "",
                "rewritten_text": None,
                "rewrite_attempted": False,
                "rewrite_failure_reason": "ControlledWriter disabled",
                # Phase 15.8 Commit 1
                "selection": None,
                "final_text": result.scene_text or "",
                "selected_source": "original",
                "selection_reason": "controlled_writer_disabled",
                "validation_original": None,
                "validation_rewritten": None,
            }
            logger.critical(
                "WRITER_NODE_ARTIFACT: schema_version=1.2, events_len=%d, text_len=%d (disabled path)",
                len(result.events or []),
                len(result.scene_text or "")
            )

            patch = _build_patch(
                scene_text=result.scene_text or "",
                final_answer=result.scene_text or "",
                writer_artifact=writer_artifact,
                metadata=patch_metadata,
            )

    # ========== 10. 确保 patch 已构建 ==========
    if patch is None:
        logger.error("writer_node: patch is None, creating error patch")
        patch = _build_patch(
            scene_text="",
            final_answer="",
            metadata={"error": "patch is None"},
            error="Internal writer error",
            phase=WorkflowPhase.VALIDATING,
        )

    # ========== 11. 强制确保关键字段存在于 payload ==========
    payload = patch.to_dict()

    if planner_outputs is None:
        planner_outputs = []
    payload["planner_outputs"] = planner_outputs
    if narrative_intent is not None:
        payload["narrative_intent"] = narrative_intent
    if current_scene_plan is not None:
        payload["scene_plan"] = current_scene_plan

    if payload.get("metadata") is None:
        payload["metadata"] = {}
    payload["metadata"]["planner_outputs"] = planner_outputs
    if narrative_intent is not None:
        payload["metadata"]["narrative_intent"] = narrative_intent.to_dict() if hasattr(narrative_intent, 'to_dict') else narrative_intent
    if current_scene_plan is not None:
        payload["metadata"]["current_scene_plan"] = current_scene_plan

    # ========== 12. 审计日志 ==========
    logger.info(
        f"writer_node final payload audit: "
        f"planner_outputs count={len(payload.get('planner_outputs', []))}, "
        f"has_narrative_intent={'narrative_intent' in payload}, "
        f"has_scene_plan={'scene_plan' in payload}, "
        f"metadata keys={list(payload.get('metadata', {}).keys())}"
    )

    try:
        import json
        payload_str = json.dumps(payload, default=str, ensure_ascii=False)[:500]
        logger.info(f"writer_node payload snippet: {payload_str}...")
    except Exception:
        pass

    return payload


async def validate_node(state: AgentState, runtime: WriterRuntime) -> dict:
    """
    Validator 节点 - 使用 Runtime 注入的 ValidationPolicy 控制行为。

    Phase 15.7-B1: 增加 Rewritten 双轨观察，但不改变生产文本。
    """
    # ========== Phase 15.5: 获取 writer_artifact ==========
    writer_artifact = getattr(state, "writer_artifact", None)
    if writer_artifact is None:
        writer_artifact = {}
        state.writer_artifact = writer_artifact

    # ========== 强制恢复 planner_outputs 等（原有逻辑） ==========
    logger.info(f"[validate_node] FULL STATE: planner_outputs={state.planner_outputs}, metadata keys={list(state.metadata.keys())}")
    logger.info(f"[validate_node] state.planner_outputs type: {type(state.planner_outputs)}, length: {len(state.planner_outputs) if state.planner_outputs else 0}")
    logger.info(f"[validate_node] state.metadata.get('planner_outputs') type: {type(state.metadata.get('planner_outputs'))}, length: {len(state.metadata.get('planner_outputs', []))}")

    planner_outputs = state.planner_outputs
    if planner_outputs is None:
        logger.error("validate_node: planner_outputs is None (missing state)")
        return StatePatch(error="planner_outputs missing from state").to_dict()
    if not planner_outputs:
        logger.warning("validate_node: planner_outputs is empty, validation may be incomplete")

    # 从 planner_outputs 重建 narrative_intent（如果状态中缺失）
    if state.narrative_intent is None and planner_outputs:
        try:
            first_output = planner_outputs[0]
            if first_output and "narrative_intent" in first_output:
                from src.writing.narrative_intent import NarrativeIntent
                intent_data = first_output["narrative_intent"]
                if isinstance(intent_data, dict):
                    state.narrative_intent = NarrativeIntent.from_dict(intent_data)
                    logger.info(f"validate_node: reconstructed narrative_intent from planner_outputs[0]")
        except Exception as e:
            logger.warning(f"validate_node: failed to reconstruct narrative_intent: {e}")

    logger.info(f"[validate_node] state.scene_text type: {type(state.scene_text)}")
    logger.info(f"[validate_node] state.scene_text length: {len(state.scene_text) if state.scene_text else 0}")
    logger.info(f"[validate_node] state.scene_text first 200: {state.scene_text[:200] if state.scene_text else 'None'}")

    if state.compressed_state:
        if isinstance(state.compressed_state, dict):
            if "recent_scene_roles" in state.compressed_state:
                state.metadata["recent_scene_roles"] = state.compressed_state["recent_scene_roles"]
        elif hasattr(state.compressed_state, 'recent_scene_roles'):
            state.metadata["recent_scene_roles"] = state.compressed_state.recent_scene_roles

    state.validation_mode = "novel"

    logger.info(f"🔍 state.metadata.get('active_loop'): {state.metadata.get('active_loop')}")

    # 恢复 scene_plan
    if state.scene_plan is None:
        if state.metadata.get("current_scene_plan"):
            state.scene_plan = state.metadata["current_scene_plan"]
            logger.info("[validate_node] Restored scene_plan from metadata.current_scene_plan")
        else:
            current_idx = state.current_scene_index if state.current_scene_index is not None else 0
            if state.scene_plan_list and current_idx < len(state.scene_plan_list):
                state.scene_plan = state.scene_plan_list[current_idx]
                state.metadata["current_scene_plan"] = state.scene_plan
                logger.info(f"[validate_node] Restored scene_plan from scene_plan_list[{current_idx}]")
            else:
                logger.warning(f"[validate_node] No scene_plan available for scene {current_idx}")

    if not hasattr(state, 'planning_contract') or state.planning_contract is None:
        if state.metadata and "planning_contract" in state.metadata:
            state.planning_contract = state.metadata["planning_contract"]
            scene_id = state.planning_contract.get('scene_id') if state.planning_contract else 'None'
            logger.info(f"✅ Restored Planning Contract from metadata: {scene_id}")

    scene_plan = state.scene_plan
    if state.planning_contract is None:
        if scene_plan and "planning_contract" in scene_plan:
            state.planning_contract = scene_plan["planning_contract"]
            scene_id = state.planning_contract.get('scene_id') if state.planning_contract else 'None'
            logger.info(f"✅ Loaded Planning Contract for validation: {scene_id}")
        else:
            logger.warning("⚠️ No planning_contract found in scene_plan")

    # ========== 获取 scene_role ==========
    scene_role = None
    if state.narrative_intent:
        scene_role = state.narrative_intent.scene_role.value if hasattr(state.narrative_intent.scene_role, 'value') else state.narrative_intent.scene_role
        logger.info(f"✅ 从 narrative_intent 获取 scene_role: {scene_role}")
    elif state.scene_plan:
        scene_role = state.scene_plan.get("scene_role")
    elif state.metadata.get("current_scene_plan"):
        scene_role = state.metadata["current_scene_plan"].get("scene_role")
    elif state.metadata.get("scene_role"):
        scene_role = state.metadata.get("scene_role")

    if scene_role:
        recent_roles = state.metadata.get("recent_scene_roles", [])
        recent_roles.append(scene_role)
        if len(recent_roles) > 20:
            recent_roles = recent_roles[-20:]
        state.metadata["recent_scene_roles"] = recent_roles
        logger.info(f"✅ 记录 scene_role: {scene_role} (总数 {len(recent_roles)})")
    else:
        logger.warning(f"No scene_role found for scene {state.current_scene_index}")

    # ========== 1. 验证 ==========
    # 创建 Validator（如果当前上下文没有）
    # 注意：如果 validate_node 已有 validator 变量，直接复用。
    # 此处假设尚未创建，按需创建。
    validator = ValidatorAgent()
    
    # 保存原始 scene_text 以便后续比较
    original_scene_text = state.scene_text

    # ---- 1.1 原有生产验证（Original） ----
    updates = await validator.run(state)
    validation_result = updates.get("validation_result", {})

    # ========== B2-2C2: 生产救援 ==========
    from src.writing.validation_v2.production_bridge import get_b2_2_bridge

    if not validation_result.get("passed", False):
        try:
            bridge = get_b2_2_bridge()

            contract_data = state.planning_contract
            # C2-3: writer_events 可以为空，scene_text 作为 fallback
            writer_events = (
                writer_artifact.get("events", []) if writer_artifact else []
            )
            scene_text = (
                writer_artifact.get("scene_text", "")
                if writer_artifact
                else state.scene_text
            )

            # C2-3: 只要 contract_data 存在就运行，不要求 writer_events 非空
            if contract_data:
                # ============================================================
                # C3.4.2d: 构造 audit_context（观测层，不影响判定）
                # 构造失败时 audit_ctx=None，bridge 会完全跳过 audit
                # ============================================================
                try:
                    audit_ctx = AuditContext(
                        novel_id=state.novel_id or "",
                        volume_num=state.current_volume,
                        chapter_num=state.current_chapter,
                        scene_idx=(
                            state.current_scene_index
                            if state.current_scene_index is not None
                            else 0
                        ),
                        scene_id=(
                            state.scene_plan.get("scene_id")
                            if state.scene_plan
                            else None
                        ),
                        source="production",
                        mode="production",  # bridge 内部按 claim_type 覆盖
                    )
                except Exception as _e:
                    logger.warning(f"[Audit] Failed to build context: {_e}")
                    audit_ctx = None
                # ============================================================
                # C3.4.3B.3: 构造 bridge_audit_context
                # 观测层，不影响判定；构造失败时为 None，bridge 会跳过 audit
                # ============================================================
                try:
                    bridge_audit_ctx = BridgeAuditContext(
                        novel_id=state.novel_id or "",
                        volume_num=state.current_volume,
                        chapter_num=state.current_chapter,
                        scene_idx=(
                            state.current_scene_index
                            if state.current_scene_index is not None
                            else 0
                        ),
                        scene_id=(
                            state.scene_plan.get("scene_id")
                            if state.scene_plan
                            else None
                        ),
                    )
                except Exception as _e:
                    logger.warning(
                        f"[BridgeAudit] Failed to build context: {_e}"
                    )
                    bridge_audit_ctx = None
                # ============================================================

                b2_2_result = await bridge.try_rescue(
                    contract=contract_data,
                    writer_events=writer_events,
                    scene_text=scene_text,
                    original_validation_result=validation_result,
                    scene_id=(
                        state.scene_plan.get("scene_id", "unknown")
                        if state.scene_plan
                        else "unknown"
                    ),
                    audit_context=audit_ctx,                 # C3.4.2d
                    bridge_audit_context=bridge_audit_ctx,   # C3.4.3B.3
                )
                # C2-4: 双轨审计，包含完整 evidence_ids
                state.metadata["b2_2_bridge"] = {
                    "triggered": b2_2_result.triggered,
                    "verdict": b2_2_result.verdict,
                    "confidence": b2_2_result.confidence,
                    "rescued": b2_2_result.rescued,
                    "claim_type": b2_2_result.claim_type,
                    "reason": b2_2_result.reason,
                    "claim_id": b2_2_result.claim_id,
                    "evidence_ids": b2_2_result.evidence_ids,  # C2-4
                }

                if b2_2_result.rescued:
                    logger.info(
                        f"[B2-2C2] Rescue applied for "
                        f"{state.scene_plan.get('scene_id', 'unknown')}: "
                        f"confidence={b2_2_result.confidence}, "
                        f"claim_id={b2_2_result.claim_id}, "
                        f"evidence_count={len(b2_2_result.evidence_ids)}"
                    )
                    validation_result["passed"] = True
                    validation_result["b2_2_rescued"] = True
                    validation_result["b2_2_confidence"] = b2_2_result.confidence
                    validation_result["b2_2_reason"] = b2_2_result.reason
                    validation_result["b2_2_evidence_ids"] = b2_2_result.evidence_ids

        except Exception as e:
            logger.error(f"[B2-2C2] Bridge error: {e}", exc_info=True)
    # =============================================================

    # ---- 1.2 记录 Original 结果到 artifact ----
    if writer_artifact:
        writer_artifact["validation_original"] = {
            "passed": validation_result.get("passed", False),
            "feedback": validation_result.get("feedback", ""),
        }
        # 确保 original_text 已设置
        if "original_text" not in writer_artifact or not writer_artifact["original_text"]:
            writer_artifact["original_text"] = original_scene_text

    # ---- 1.3 额外验证 Rewritten（仅当存在且成功） ----
    if writer_artifact and writer_artifact.get("rewrite_attempted", False):
        rewritten_text = writer_artifact.get("rewritten_text")
        if rewritten_text and len(rewritten_text.strip()) > 50:
            # ========== 修复：不深拷贝整个 state，只构造轻量级 AgentState ==========
            from src.orchestrator.state import AgentState
            temp_state_rew = AgentState(
                user_input=getattr(state, 'user_input', ''),
                novel_id=getattr(state, 'novel_id', None),
                current_volume=getattr(state, 'current_volume', 1),
                current_chapter=getattr(state, 'current_chapter', 1),
                current_scene_index=getattr(state, 'current_scene_index', 0),
                current_state=getattr(state, 'current_state', {}),
                scene_text=rewritten_text,
                scene_plan=getattr(state, 'scene_plan', None),
                scene_plan_list=getattr(state, 'scene_plan_list', []),
                planning_contract=getattr(state, 'planning_contract', None),
                narrative_intent=getattr(state, 'narrative_intent', None),
                compressed_state=getattr(state, 'compressed_state', None),
                metadata={
                    "active_loop": state.metadata.get("active_loop") if hasattr(state, 'metadata') else None,
                    "planning_contract": state.metadata.get("planning_contract") if hasattr(state, 'metadata') else None,
                    "current_scene_plan": state.metadata.get("current_scene_plan") if hasattr(state, 'metadata') else None,
                    "recent_scene_roles": state.metadata.get("recent_scene_roles") if hasattr(state, 'metadata') else [],
                },
                validation_mode="novel",
            )
            rewritten_updates = await validator.run(temp_state_rew)
            # ===========================================================
            # ========== 保存 Rewritten 验证结果 ==========
            rewritten_result = rewritten_updates.get("validation_result", {})
            writer_artifact["validation_rewritten"] = {
                "passed": rewritten_result.get("passed", False),
                "feedback": rewritten_result.get("feedback", ""),
            }
            logger.info(
                f"[15.7-B1] Dual observation: "
                f"original_pass={validation_result.get('passed', False)}, "
                f"rewritten_pass={rewritten_result.get('passed', False)}"
            )
            # =====================================
        else:
            writer_artifact["validation_rewritten"] = None
    else:
        writer_artifact["validation_rewritten"] = None
        
    # ---- 1.4 B1 数据契约（硬不变量） ----
    if writer_artifact:
        # final_text 永远指向 original_text
        writer_artifact["final_text"] = writer_artifact.get("original_text", original_scene_text)
        # selected_source 仅允许观察值
        if writer_artifact.get("rewrite_attempted", False):
            writer_artifact["selected_source"] = "original_observational"
        else:
            writer_artifact["selected_source"] = "original_no_rewrite"
        
        # 同步回 state
        state.writer_artifact = writer_artifact

    # ---- 1.5 生产文本锁定 ----
    # 无论 Rewrite 结果如何，state.scene_text 始终为 Original
    state.scene_text = original_scene_text
    state.final_answer = original_scene_text

    # ========== 2. 后续原有业务逻辑（SceneCompletionService 等） ==========
    # 注意：以下代码复用自原有 validate_node，仅作示意，实际应保持原样。
    # 此段保留原有实现，但确保使用 state.scene_text（已锁定为 Original）

    # ... 原有的 ValidationPolicy、重试逻辑、SceneCompletionService 调用等 ...

    # 最终返回 StatePatch
    # ============================================================
    # Phase 15.7-B2-0: No-op Selection Observation
    # ============================================================
    # ============================================================
    # Phase 15.8 Commit 1: Selection Observation
    # ============================================================
    if not isinstance(writer_artifact, dict):
        writer_artifact = {}

    original_passed = writer_artifact.get("validation_original", {}).get("passed", False)
    validation_rewritten = writer_artifact.get("validation_rewritten")
    rewritten_passed = False if validation_rewritten is None else validation_rewritten.get("passed", False)
    rewritten_exists = bool(writer_artifact.get("rewritten_text"))

    # 从 ControlledWriter 的 selection 契约读取；缺失则回退到 original 语义
    selection = writer_artifact.get("selection") or {}
    selected_source = selection.get("selected_source") or "original"
    selection_reason = selection.get("selection_reason") or "missing_selection"

    # Commit 1 硬不变量防御：若上游异常返回 rewritten，强制 original
    if selected_source != "original":
        logger.critical(
            "[15.8-C1] UNEXPECTED selected_source=%s in Commit 1, forcing original",
            selected_source,
        )
        selected_source = "original"
        selection_reason = "commit1_invariant_forced"

    writer_artifact["selected_source"] = selected_source
    writer_artifact["selection_reason"] = selection_reason
    writer_artifact["structural_safe"] = selection.get("structural_safe", False)
    writer_artifact["rewrite_available"] = selection.get("rewrite_available", False)
    writer_artifact["b2_phase"] = "B2-1"

    state.metadata["selected_source"] = selected_source
    state.metadata["selection_reason"] = selection_reason
    state.metadata["structural_safe"] = writer_artifact["structural_safe"]
    state.metadata["rewrite_available"] = writer_artifact["rewrite_available"]
    state.metadata["b2_phase"] = "B2-1"
    state.metadata["original_passed_at_selection"] = original_passed
    state.metadata["rewritten_passed_at_selection"] = rewritten_passed
    state.metadata["rewritten_exists"] = rewritten_exists

    logger.info(
        "[15.8-C1] Selection observation: "
        "selected_source=%s, selection_reason=%s, "
        "structural_safe=%s, rewrite_available=%s, "
        "original_passed=%s, rewritten_passed=%s, rewritten_exists=%s",
        selected_source,
        selection_reason,
        writer_artifact["structural_safe"],
        writer_artifact["rewrite_available"],
        original_passed,
        rewritten_passed,
        rewritten_exists,
    )
    # ============================================================
    # ============================================================
    # 强制写入 Rewrite 文本到 shadow 目录（独立于 _run_shadow_rewrite）
    # ============================================================
    if writer_artifact and writer_artifact.get("rewritten_text"):
        rewritten_text = writer_artifact["rewritten_text"]
        if rewritten_text and state.novel_id:
            try:
                from pathlib import Path
                shadow_dir = Path(f"data/novels/{state.novel_id}/shadow/vol_{state.current_volume:03d}")
                shadow_dir.mkdir(parents=True, exist_ok=True)
                chapter_file = shadow_dir / f"chap_{state.current_chapter:03d}.txt"
                scene_marker = f"\n\n<!-- scene {state.current_scene_index:02d} -->\n\n"
                if not chapter_file.exists():
                    chapter_file.write_text(rewritten_text, encoding="utf-8")
                else:
                    with open(chapter_file, "a", encoding="utf-8") as f:
                        f.write(scene_marker)
                        f.write(rewritten_text)
                logger.info(f"[B2-0] Shadow write forced: {chapter_file} (scene {state.current_scene_index})")
            except Exception as e:
                logger.error(f"[B2-0] Force shadow write failed: {e}", exc_info=True)

    # ============================================================
    # 场景完成：推进 writing_progress（无论验证是否通过）
    # ============================================================
    from src.writing.services import SceneCompletionService, SceneCompletionCommand

    scene_plan = state.scene_plan or {}
    scene_idx = state.current_scene_index if state.current_scene_index is not None else 0
    total_scenes = state.total_scenes_in_chapter

    parsed_output = {
        "scene_text": writer_artifact.get("scene_text", ""),
        "events": writer_artifact.get("events", []),
        "foreshadowing": writer_artifact.get("foreshadowing", []),
    }

    current_world = WorldState.from_dict(state.current_state) if state.current_state else WorldState()

    cmd = SceneCompletionCommand(
        novel_id=state.novel_id,
        volume=state.current_volume,
        chapter=state.current_chapter,
        scene_idx=scene_idx,
        total_scenes=total_scenes,
        current_world_state=current_world.to_dict(),
        parsed_output=parsed_output,
        scene_plan=scene_plan,
        character_intents=state.metadata.get("character_intents"),
        voice_memory=state.metadata.get("voice_fingerprint"),
        raw_output=state.scene_text,
        narrative_intent=state.narrative_intent,
        validation_passed=validation_result.get("passed", False),  # ← 新增
    )

    completion_result = await SceneCompletionService.execute(cmd)
    if completion_result.error:
        logger.error(f"SceneCompletion failed: {completion_result.error}")
        completion_patch = StatePatch(
            current_scene_index=scene_idx + 1,
            phase=WorkflowPhase.VALIDATING,
        )
    else:
        completion_patch = completion_result.state_patch
        logger.info(f"SceneCompletion succeeded, chapter_finished={completion_result.chapter_finished}")

    # ============================================================
    # 构造 StatePatch（合并完成状态）
    # ============================================================
    patch = StatePatch(
        scene_text=state.scene_text,
        final_answer=state.final_answer,
        validation_result=validation_result,
        writer_artifact=writer_artifact,
        metadata=state.metadata,
        current_scene_index=completion_patch.current_scene_index if completion_patch else scene_idx + 1,
        current_chapter=completion_patch.current_chapter if completion_patch else state.current_chapter,
        current_volume=completion_patch.current_volume if completion_patch else state.current_volume,
        phase=completion_patch.phase if completion_patch else WorkflowPhase.VALIDATING,
    )
    return patch.to_dict()


async def research_node(state: AgentState) -> dict[str, Any]:
    ra = ResearchAgent()
    result = await ra.run(state)
    return {
        "research_results": result.get("research_results", []),
        "sources": result.get("sources", []),
        "current_node": "research"
    }


async def code_node(state: AgentState) -> dict[str, Any]:
    ea = ExecutorAgent()
    updates = await ea.run(state)
    return {
        "code_generated": updates.get("code_generated", ""),
        "code_file_path": updates.get("code_file_path", ""),
        "execution_result": updates.get("execution_result"),
        "current_node": "code"
    }


async def scheduler_node(state: AgentState) -> dict[str, Any]:
    return {"plan_status": "no_plan", "subtask_results": {}, "current_node": "scheduler"}


def advance_subtask_node(state: AgentState) -> dict[str, Any]:
    return {"subtasks": []}


async def tool_node_v2(state: AgentState) -> dict[str, Any]:
    return {"pending_tool_calls": [], "tool_results": [], "current_node": "tool_node"}

async def _update_scene_plan_drama(novel_id: str, volume: int, chapter: int, scene_idx: int, drama_struct: dict):
    """更新 scene_execution_units 中的 plan_json，加入 drama 字段"""
    pool = get_db_pool()
    if not pool:
        return
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT plan_json FROM scene_execution_units WHERE novel_id=$1 AND volume_num=$2 AND chapter_num=$3 AND scene_index=$4",
            novel_id, volume, chapter, scene_idx
        )
        if row:
            plan = json.loads(row["plan_json"])
            plan["drama"] = drama_struct
            await conn.execute(
                "UPDATE scene_execution_units SET plan_json=$1 WHERE novel_id=$2 AND volume_num=$3 AND chapter_num=$4 AND scene_index=$5",
                json.dumps(plan, ensure_ascii=False), novel_id, volume, chapter, scene_idx
            )
            logger.info(f"Updated drama in scene plan for scene {scene_idx}")
            

async def rewrite_node(state: AgentState) -> dict:
    """Rewrite 节点：调用 RewriteAgent 进行戏剧放大"""
    from src.agents.rewrite import RewriteAgent
    agent = RewriteAgent()
    return await agent.run(state)


async def drama_planner_node(state: AgentState) -> dict:
    """Drama Planner 节点：生成戏剧结构"""
    from src.agents.drama_planner import DramaPlannerAgent
    agent = DramaPlannerAgent()
    return await agent.run(state)