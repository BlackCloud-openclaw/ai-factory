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
from src.writing.world_state import WorldState
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


# 全局 logger
logger = setup_logging("orchestrator.nodes")

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
    """跳过当前场景：更新 writing_progress 和 scene_execution_units 状态"""
    pool = get_db_pool()
    if not pool:
        return
    async with pool.acquire() as conn:
        async with conn.transaction():
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


# src/orchestrator/nodes.py

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
    # 可选：打印 intent_id 列表
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
    logger.info("WritingAgent starting")

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
    # 1. 获取 scene_plan_list（必须存在）
    scene_plan_list = state.scene_plan_list
    if scene_plan_list is None:
        logger.error("writer_node: scene_plan_list is None (missing state)")
        return StatePatch(error="scene_plan_list missing from state").to_dict()
    if not scene_plan_list:
        logger.error("writer_node: scene_plan_list is empty")
        return StatePatch(error="scene_plan_list empty").to_dict()

    if not scene_plan_list:
        logger.error(f"writer_node: scene_plan_list is empty for chapter {state.current_chapter}")
        return StatePatch(error="Scene plan list empty").to_dict()

    if current_idx >= len(scene_plan_list):
        logger.error(f"writer_node: invalid scene index {current_idx} (len={len(scene_plan_list)})")
        return StatePatch(error="Invalid scene index").to_dict()

    current_scene_plan = scene_plan_list[current_idx]
    state.scene_plan = current_scene_plan
    state.metadata["current_scene_plan"] = current_scene_plan  # 备份给 validate 使用
    # ======================================================

    # ========== 4. 提取 Planning Contract ==========
    planning_contract = current_scene_plan.get("planning_contract")
    if planning_contract:
        state.planning_contract = planning_contract        
        scene_id = planning_contract.get('scene_id', 'unknown')
        logger.info(f"✅ 从 scene_plan 提取 Planning Contract: {scene_id}")
    else:
        logger.warning(f"⚠️ scene_plan 中无 planning_contract (scene_idx={current_idx})")

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
                # 从场景计划中提取信息
                intent_data = scene.get("narrative_intent") or {}
                scene_role_str = intent_data.get("scene_role", "transition")
                try:
                    scene_role = SceneRole(scene_role_str)
                except ValueError:
                    scene_role = SceneRole.TRANSITION
                
                # 构建 NarrativeIntent
                narrative_intent = NarrativeIntent(
                    intent_id=intent_data.get("intent_id", f"rebuilt_{idx}"),
                    scene_role=scene_role,
                    objective=intent_data.get("objective", scene.get("goal", "推进剧情")),
                    preconditions=[],
                    beats=[],
                    consequences=[],
                    interaction_plan=None,
                )
                
                # 构建 ExecutionContract
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
                    # 如果没有 contract，使用默认空 contract
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

    # 继续正常逻辑...
    # 注意：后续代码将使用 planner_outputs，所以必须保证它非空
    # 但这里继续往下走，因为已确保 planner_outputs 有效

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
        else:
            logger.warning(f"writer_node: planner_outputs[{current_idx}] 缺少 narrative_intent")
    else:
        logger.warning(
            f"writer_node: planner_outputs empty or index out of range "
            f"(len={len(planner_outputs)}, idx={current_idx})"
        )

    logger.info(
        logger.info(f"writer_node input planner_outputs count={len(planner_outputs)}")
    )
    # ================================================================

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

    # 优先使用 state.narrative_intent（已从 planner_outputs 恢复）
    narrative_intent = state.narrative_intent

    # 构建 WritingGoal
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

    # 如果 planning_contract 是字典，转为对象
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
        narrative_intent=narrative_intent,
        constraints=constraints,
        writing_goal=writing_goal,
        execution_contract=planning_contract_obj,   # ✅ 新增
    )

    cw = ControlledWriter(runtime_services=runtime.runtime_services)

    # ========== 8. 辅助函数：构建包含关键状态的 StatePatch ==========
    def _build_patch(
        scene_text: str,
        final_answer: str,
        metadata: Dict[str, Any] = None,
        error: str = None,
        phase: WorkflowPhase = None,
        writer_artifact: Optional[Dict[str, Any]] = None,  # 新增
    ) -> StatePatch:
        """构建包含关键状态的 StatePatch，确保下游节点能获取 planner_outputs 等"""
        # 构造 metadata（如果未提供则创建）
        patch_metadata = metadata or {}
        # 确保 planner_outputs、narrative_intent、scene_plan 在 metadata 中
        if planner_outputs:
            patch_metadata["planner_outputs"] = planner_outputs
        if narrative_intent:
            patch_metadata["narrative_intent"] = narrative_intent.to_dict() if hasattr(narrative_intent, 'to_dict') else narrative_intent
        if current_scene_plan:
            patch_metadata["current_scene_plan"] = current_scene_plan

        # 创建 StatePatch，显式设置字段
        patch = StatePatch(
            scene_text=scene_text,
            final_answer=final_answer,
            planner_outputs=planner_outputs,
            narrative_intent=narrative_intent,
            scene_plan=current_scene_plan,
            metadata=patch_metadata,
            writer_artifact=writer_artifact,  # 新增传递
        )
        if error:
            patch.error = error
        if phase:
            patch.phase = phase
        return patch

    # ========== 9. 执行写入 ==========
    patch = None
    if getattr(config, 'controlled_writer_enabled', True):
        exec_units = []
        if planning_contract and isinstance(planning_contract, dict):
            exec_units = planning_contract.get("execution", {}).get("units", [])

        if len(exec_units) >= 3:
            logger.info(f"🚀 使用 ControlledWriter: {len(exec_units)} 个执行单元")
            try:
                result = await cw.execute(writing_contract)
                if result.text:
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

                    # ============================================================
                    # Commit D.2: 构造 Writer Artifact (ControlledWriter 路径)
                    # ============================================================
                    writer_artifact = {
                        "schema_version": "1.0",
                        "scene_text": result.text,
                        "events": result.events,
                        "foreshadowing": [],
                    }
                    # ============================================================

                    # ========== D.3 观测点 4：Node 层 Artifact ==========
                    logger.critical(
                        "WRITER_NODE_ARTIFACT: events_len=%d, text_len=%d",
                        len(result.events),
                        len(result.text)
                    )
                    # =================================================

                    patch = _build_patch(
                        scene_text=result.text,
                        final_answer=result.text,
                        writer_artifact=writer_artifact,
                        metadata=patch_metadata,
                    )
                else:
                    logger.error("❌ ControlledWriter 返回空文本")
                    patch = _build_patch(
                        scene_text="",
                        final_answer="",
                        metadata={"error": "ControlledWriter 返回空结果"},
                        error="ControlledWriter 返回空结果",
                        phase=WorkflowPhase.VALIDATING,
                    )
            except Exception as e:
                logger.exception(f"❌ ControlledWriter 执行失败: {e}")
                patch = _build_patch(
                    scene_text="",
                    final_answer="",
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
                # D.4.1-a: 显式传递 planning_contract
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

                # ========== D.3: WritingService 路径补丁 ==========
                writer_artifact = {
                    "schema_version": "1.0",
                    "scene_text": result.scene_text or "",
                    "events": result.events or [],
                    "foreshadowing": [],
                }
                logger.critical(
                    "WRITER_NODE_ARTIFACT: events_len=%d, text_len=%d",
                    len(result.events or []),
                    len(result.scene_text or "")
                )
                # =================================================

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
            # D.4.1-a: 显式传递 planning_contract
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

            # ========== D.3: WritingService 路径补丁 ==========
            writer_artifact = {
                "schema_version": "1.0",
                "scene_text": result.scene_text or "",
                "events": result.events or [],
                "foreshadowing": [],
            }
            logger.critical(
                "WRITER_NODE_ARTIFACT: events_len=%d, text_len=%d",
                len(result.events or []),
                len(result.scene_text or "")
            )
            # =================================================

            patch = _build_patch(
                scene_text=result.scene_text or "",
                final_answer=result.scene_text or "",
                writer_artifact=writer_artifact,
                metadata=patch_metadata,
            )

    # ========== 10. 确保 patch 已构建（防御） ==========
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

    # 确保 planner_outputs 是列表（不是 None）
    if planner_outputs is None:
        planner_outputs = []
    # 确保 payload 顶级字段包含 planner_outputs
    payload["planner_outputs"] = planner_outputs
    if narrative_intent is not None:
        payload["narrative_intent"] = narrative_intent
    if current_scene_plan is not None:
        payload["scene_plan"] = current_scene_plan

    # 确保 metadata 存在并包含关键数据
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

    # 打印完整 payload 的前 500 字符（用于调试）
    try:
        import json
        payload_str = json.dumps(payload, default=str, ensure_ascii=False)[:500]
        logger.info(f"writer_node payload snippet: {payload_str}...")
    except Exception:
        pass

    return payload
# ============================================================================
# validate_node 完整函数（增加详细日志和从 metadata 恢复的逻辑）
# ============================================================================
# src/orchestrator/nodes.py

async def validate_node(state: AgentState, runtime: WriterRuntime) -> dict:
    """
    Validator 节点 - 使用 Runtime 注入的 ValidationPolicy 控制行为。
    """
    # ========== 强制恢复：从 metadata 中提取 planner_outputs 和 narrative_intent ==========
    logger.info(f"[validate_node] FULL STATE: planner_outputs={state.planner_outputs}, metadata keys={list(state.metadata.keys())}")
    logger.info(f"[validate_node] state.planner_outputs type: {type(state.planner_outputs)}, length: {len(state.planner_outputs) if state.planner_outputs else 0}")
    logger.info(f"[validate_node] state.metadata.get('planner_outputs') type: {type(state.metadata.get('planner_outputs'))}, length: {len(state.metadata.get('planner_outputs', []))}")

    # ===== 强类型读取 planner_outputs =====
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

    # 从 compressed_state 恢复 recent_scene_roles
    if state.compressed_state:
        if isinstance(state.compressed_state, dict):
            if "recent_scene_roles" in state.compressed_state:
                state.metadata["recent_scene_roles"] = state.compressed_state["recent_scene_roles"]
        elif hasattr(state.compressed_state, 'recent_scene_roles'):
            state.metadata["recent_scene_roles"] = state.compressed_state.recent_scene_roles

    state.validation_mode = "novel"

    logger.info(f"🔍 state.metadata.get('active_loop'): {state.metadata.get('active_loop')}")

    # ========== 恢复 scene_plan ==========
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

    # 恢复 planning_contract
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
    validator = ValidatorAgent()
    updates = await validator.run(state)
    validation_result = updates.get("validation_result", {})
    
    # D.5.2: FeedbackCompiler preview (no retry yet)
    if hasattr(validation_result, "missing_changes"):
        missing_changes = validation_result.missing_changes
    else:
        missing_changes = validation_result.get("missing_changes", [])

    if missing_changes:
        from src.writing.validation.feedback import ValidationFeedbackCompiler
        compiler = ValidationFeedbackCompiler(max_items=3)
        feedback = compiler.compile(missing_changes)
        scene_id = state.scene_plan.get("scene_id", "unknown") if state.scene_plan else "unknown"
        logger.info(
            "VALIDATION_FEEDBACK_PREVIEW: scene=%s chars=%d preview=%s",
            scene_id,
            len(feedback),
            feedback[:200]
        )
        state.metadata["validation_feedback_preview"] = feedback
    else:
        logger.debug("No missing_changes, skipping feedback compilation")
    
    passed = validation_result.get("passed", False)
    should_retry = validation_result.get("should_retry", False)

    # ========== Phase 14.0C-2: 使用 ValidationPolicy ==========
    policy = runtime.validation_policy
    # ========== 2. 验证失败处理 ==========
    # ========== Phase 14.0C-2/3A: 基于 ValidatorOutput 状态判断 ==========
    validator_output = validation_result.get("validator_output")
    if validator_output is None:
        # Validator 没有输出，这是致命错误
        logger.error("Validator returned no output, cannot proceed")
        patch = StatePatch(
            error="Validator failed without output",
            validation_result=validation_result,
            phase=WorkflowPhase.VALIDATING,
        )
        return patch.to_dict()

    status = validator_output.get("status")
    violations = validator_output.get("violations", [])

    if status == "failed":
        # 检查是否允许降级通过（开发环境）
        if policy.allow_degraded_pass and not policy.fail_on_error:
            # 降级通过
            state.metadata["validation_degraded"] = True
            logger.warning(f"Scene {state.current_scene_index} validation failed, but degraded pass allowed (bypass retry).")
            # 获取 parsed_output
            parsed_output = validation_result.get("parsed_output", {})
            if not parsed_output or not parsed_output.get("scene_text"):
                # 尝试从 state.scene_text 构造
                if state.scene_text:
                    parsed_output = {"scene_text": state.scene_text, "events": [], "_source": "degraded_fallback"}
                    logger.warning("Constructed parsed_output from state.scene_text for degraded pass")
                else:
                    logger.error("Degraded pass but no parsed_output and no scene_text")
                    return StatePatch(
                        error="Degraded pass but no parsed_output",
                        phase=WorkflowPhase.VALIDATING,
                        validation_result=validation_result,
                    ).to_dict()
            # 标记验证通过（以便后续逻辑继续）
            passed = True
            # 跳过重试，继续执行后面的逻辑
        else:
            # ============================================================
            # D.5.3: Contract Retry Controller
            # 仅当 missing_changes 非空时介入，覆盖 should_retry 和 feedback
            # ============================================================
            contract_retry_feedback = None  # 标记 Controller 是否提供了 feedback

            if missing_changes:
                from src.writing.validation.retry_controller import ContractRetryController
                from src.writing.runtime.enforcement_mode import EnforcementMode
                from src.writing.runtime.validation_policy import ValidationPolicy

                # 确保 policy 有 enforcement_mode（若无则用默认 OBSERVE）
                if not hasattr(policy, 'enforcement_mode') or policy.enforcement_mode is None:
                    policy = ValidationPolicy(
                        allow_degraded_pass=policy.allow_degraded_pass,
                        max_retry=policy.max_retry,
                        fail_on_error=policy.fail_on_error,
                        recovery_enabled=policy.recovery_enabled,
                        enforcement_mode=EnforcementMode.OBSERVE,
                        #enforcement_mode=EnforcementMode.RETRY,  # <-- 改为 RETRY
                    )

                retry_controller = ContractRetryController()
                retry_decision = retry_controller.decide(
                    missing_changes=missing_changes,
                    retry_count=state.retry_count,
                    policy=policy,
                )

                if retry_decision.should_retry:
                    should_retry = True
                    contract_retry_feedback = retry_decision.writing_feedback
                    validation_result["feedback"] = contract_retry_feedback
                    state.metadata["writing_feedback"] = contract_retry_feedback

                    scene_id = state.scene_plan.get("scene_id", "unknown") if state.scene_plan else "unknown"
                    logger.info(
                        "CONTRACT_RETRY_TRIGGERED: scene=%s retry=%d/%d reason=%s",
                        scene_id,
                        retry_decision.next_retry_count,
                        policy.max_retry,
                        retry_decision.reason,
                    )
                else:
                    should_retry = False

            # ============================================================
            # 原有重试逻辑（优先使用 contract_retry_feedback）
            # ============================================================
            if should_retry and state.retry_count < policy.max_retry:
                retry_count = state.retry_count + 1

                # 如果 Controller 已提供 feedback，直接使用它
                if contract_retry_feedback is not None:
                    feedback_str = contract_retry_feedback
                else:
                    # 旧逻辑：从 semantic_validation 中提取 missing_names
                    control_scores = validation_result.get("control_scores", {})
                    semantic = control_scores.get("semantic_validation", {})
                    missing_names = semantic.get("missing_names", [])
                    total_missing = len(missing_names)

                    logger.info(
                        "Contract Realization Feedback",
                        extra={
                            "total_missing": total_missing,
                            "feedback_count": min(total_missing, 3),
                            "scene_index": state.current_scene_index,
                        }
                    )

                    if missing_names:
                        missing_items = [
                            {"type": "state_change", "name": name}
                            for name in missing_names[:3]
                        ]
                        feedback = {
                            "type": "contract_realization",
                            "missing_changes": missing_items,
                            "instruction": "请在下一版生成的 events 中，通过剧情发展自然体现以上缺失的状态变化，而非机械输出字段名。"
                        }
                        feedback_str = json.dumps(feedback, ensure_ascii=False)
                    else:
                        # 若没有更具体的 missing_names，使用 validation_result["feedback"]
                        feedback_str = validation_result.get("feedback", "验证未通过，请重试。")

                logger.info(f"Scene {state.current_scene_index} validation failed, retrying ({retry_count}/{policy.max_retry})")
                return StatePatch(
                    validation_result=validation_result,
                    retry_count=retry_count,
                    needs_retry=True,
                    writing_feedback=feedback_str,
                    phase=WorkflowPhase.WRITING,
                ).to_dict()
            else:
                # 硬失败
                error_msg = f"Validator failed: {', '.join(v.get('description', '') for v in violations[:2])}"
                logger.error(f"Scene {state.current_scene_index} validation failed permanently: {error_msg}")
                return StatePatch(
                    error=error_msg,
                    phase=WorkflowPhase.VALIDATING,
                    validation_result=validation_result,
                ).to_dict()

    elif status == "degraded":
        # 降级通过：需要 policy 允许
        if not policy.allow_degraded_pass:
            error_msg = "Degraded pass not allowed by policy"
            logger.error(error_msg)
            return StatePatch(
                error=error_msg,
                phase=WorkflowPhase.VALIDATING,
                validation_result=validation_result,
            ).to_dict()
        # 允许 degraded pass
        state.metadata["validation_degraded"] = True
        parsed_output = validation_result.get("parsed_output", {})

    elif status == "passed":
        # 正常通过
        parsed_output = validation_result.get("parsed_output", {})

    else:
        # 未知状态
        logger.error(f"Unknown ValidatorOutput status: {status}")
        return StatePatch(
            error=f"Unknown validator status: {status}",
            phase=WorkflowPhase.VALIDATING,
            validation_result=validation_result,
        ).to_dict()

    # 如果 parsed_output 仍然为空，这是硬错误（不再 fallback）
    if not parsed_output or not parsed_output.get("scene_text"):
        logger.error(
            f"Validator passed/degraded but parsed_output missing. "
            f"status={status}, violations={len(violations)}"
        )
        return StatePatch(
            error="Validator returned status but no parsed_output",
            phase=WorkflowPhase.VALIDATING,
            validation_result=validation_result,
        ).to_dict()

    # ========== 4. 调用 SceneCompletionService ==========
    character_intents = state.metadata.get("character_intents")
    voice_memory = getattr(state, 'voice_memory', None)

    cmd = SceneCompletionCommand(
        novel_id=state.novel_id,
        volume=state.current_volume,
        chapter=state.current_chapter,
        scene_idx=state.current_scene_index,
        total_scenes=state.total_scenes_in_chapter,
        current_world_state=state.current_state,
        parsed_output=parsed_output,
        scene_plan=state.scene_plan,
        character_intents=character_intents,
        voice_memory=voice_memory,
        raw_output=state.scene_text,
        narrative_intent=state.narrative_intent,
    )

    logger.info(f"DEBUG: parsed_output keys = {list(parsed_output.keys())}, has scene_text = {'scene_text' in parsed_output}, length = {len(parsed_output.get('scene_text', ''))}")

    result = await SceneCompletionService.execute(cmd)
    logger.info(f"validate_node: service returned chapter_finished={result.chapter_finished}")

    # ========== 5. 更新 Loop 进度 ==========
    if result.chapter_finished and validation_result.get("passed", False):
        loop_advancement_score = validation_result.get("loop_advancement_score", 0.0)
        if loop_advancement_score > 0:
            try:
                pool = get_db_pool()
                if pool:
                    loop_store = LoopStore(pool)
                    active_loop = await loop_store.get_active_loop(state.novel_id)
                    if active_loop:
                        new_progress = min(1.0, active_loop.progress + loop_advancement_score)
                        await loop_store.update_progress(active_loop.id, new_progress)
                        logger.info(f"📈 Loop progress updated: {active_loop.progress:.0%} → {new_progress:.0%} (+{loop_advancement_score:.0%})")
                        if new_progress >= 1.0:
                            await loop_store.resolve_loop(active_loop.id)
                            logger.info(f"✅ Loop resolved: {active_loop.title}")
            except Exception as e:
                logger.error(f"Failed to update loop progress: {e}", exc_info=True)

    # ========== 6. 更新 NarrativeProjection ==========
    if result.state_patch and result.state_patch.error is None:
        try:
            from src.writing.projection_service import NarrativeProjectionService
            from src.writing.projection_updater import ProjectionUpdater
            from src.writing.events import event_from_dict

            projection_service = NarrativeProjectionService()
            previous = projection_service.load_current()

            intent = state.narrative_intent
            if intent is None:
                logger.warning("[validate_node] 无法获取 narrative_intent，跳过 Projection 更新")
            else:
                events = []
                if parsed_output and "events" in parsed_output:
                    for e in parsed_output["events"]:
                        evt_type = e.get('type')
                        if evt_type:
                            evt = event_from_dict(evt_type, e)
                            if evt:
                                events.append(evt)

                updater = ProjectionUpdater()
                new_projection = updater.update(previous, intent, events)
                projection_service.save(new_projection)
                logger.info(f"✅ 保存 Projection (version {new_projection.version})")
        except Exception as e:
            logger.error(f"[validate_node] Failed to update Projection: {e}", exc_info=True)

    # ========== 7. 章节切换与熵计算 ==========
    if result.chapter_finished:
        new_world_state = WorldState.from_dict(result.state_patch.current_state)
        compressed_state_dict = state.compressed_state or {}

        try:
            if compressed_state_dict:
                comp_state = CompressedState(**compressed_state_dict)
            else:
                comp_state = CompressedState(volume_num=state.current_volume)
        except Exception as e:
            logger.warning(f"Failed to load compressed_state for entropy calculation: {e}")
            comp_state = CompressedState(volume_num=state.current_volume)

        recent_scene_roles = state.metadata.get("recent_scene_roles", [])
        scene_role = state.metadata.get("recent_scene_roles", [])[-1] if state.metadata.get("recent_scene_roles") else None
        if scene_role:
            if not recent_scene_roles or recent_scene_roles[-1] != scene_role:
                recent_scene_roles.append(scene_role)
                if len(recent_scene_roles) > 20:
                    recent_scene_roles = recent_scene_roles[-20:]
                state.metadata["recent_scene_roles"] = recent_scene_roles
            logger.info(f"Scene role recorded: {scene_role}")
        else:
            logger.warning(f"No scene_role found for scene {state.current_scene_index}")

        recent_events = []
        try:
            pool = get_db_pool()
            if pool:
                event_store = NarrativeEventStore(pool)
                events_with_id = await event_store.get_events_since(
                    state.novel_id,
                    since_event_id=0,
                    limit=50
                )
                for _, evt in events_with_id:
                    recent_events.append({
                        "event_type": evt.type.value if hasattr(evt, 'type') else "unknown",
                        "scene_role": getattr(evt, 'scene_role', None),
                        "characters": getattr(evt, 'characters', []),
                        "new_lore": getattr(evt, 'new_lore', False),
                    })
        except Exception as e:
            logger.warning(f"Failed to load recent events for entropy: {e}")

        active_arcs = comp_state.character_arcs if hasattr(comp_state, 'character_arcs') else {}

        logger.info(f"Entropy inputs: recent_scene_roles={recent_scene_roles}, active_arcs_count={len(active_arcs)}, recent_events_count={len(recent_events)}")

        try:
            entropy_report = NarrativeEntropyCalculator.calculate_full(
                world_state=new_world_state,
                compressed_state=comp_state,
                recent_scene_roles=recent_scene_roles,
                recent_events=recent_events,
                active_arcs=active_arcs,
            )
            comp_state.local_entropy = entropy_report.local
            comp_state.arc_entropy = entropy_report.arc
            comp_state.civilization_entropy = entropy_report.civilization
            comp_state.recent_scene_roles = recent_scene_roles
            comp_state.narrative_entropy = (entropy_report.local + entropy_report.arc + entropy_report.civilization) / 3
            comp_state.entropy_history = comp_state.entropy_history[-9:] + [comp_state.narrative_entropy]

            logger.info(f"Narrative entropy for volume {state.current_volume}, chapter {state.current_chapter}: local={entropy_report.local:.3f}, arc={entropy_report.arc:.3f}, civ={entropy_report.civilization:.3f}")

            state.compressed_state = comp_state.model_dump()
        except Exception as e:
            logger.error(f"Failed to calculate narrative entropy: {e}", exc_info=True)

        # 手动保存快照
        pool = get_db_pool()
        if pool:
            snap_mgr = SnapshotManager(pool)
            event_store = NarrativeEventStore(pool)
            last_event_id = await event_store.get_last_event_id(state.novel_id)
            if last_event_id is None:
                last_event_id = 0
            await snap_mgr.save_snapshot(
                state.novel_id,
                new_world_state,
                last_event_id,
                state.current_volume,
                state.current_chapter,
                compressed_state=comp_state,
            )
            logger.info(f"Manually saved snapshot with entropy local={comp_state.local_entropy}, arc={comp_state.arc_entropy}, civ={comp_state.civilization_entropy}")

        # 章节切换
        transition_cmd = ChapterTransitionCommand(
            novel_id=state.novel_id,
            current_volume=state.current_volume,
            current_chapter=state.current_chapter,
            total_chapters_in_volume=getattr(state, 'total_chapters_in_volume', 0),
            outline=state.outline,
        )
        transition_result = await ChapterTransitionService.execute(transition_cmd)
        transition_result.state_patch.current_state = result.state_patch.current_state
        transition_result.state_patch.validation_result = result.state_patch.validation_result

        if state.compressed_state:
            state.metadata["compressed_state"] = state.compressed_state

        return transition_result.state_patch.to_dict()
    else:
        return result.state_patch.to_dict()


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