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
from src.execution.tools_registry import ToolsRegistry
from src.db import get_db_pool
from src.writing.summarizer import generate_chapter_summary
from src.writing.state_compressor import compress_current_state

# 新架构集成（保留）
from src.writing.world_state import WorldState
from src.writing.delta import StateDelta
from src.writing.event_store import NarrativeEventStore
from src.writing.snapshot import SnapshotManager
from src.writing.validators import validate_all
from src.writing.prompt_firewall import PromptFirewall
from src.writing.context_compiler import ContextCompiler
from src.writing.voiceprint import VoiceprintRegistry

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


# ====== Node functions ======
async def load_memory_node(state: AgentState) -> dict[str, Any]:
    return await _memory_agent.run(state)

async def save_memory_node(state: AgentState) -> dict[str, Any]:
    logger.info(f"Save memory for {state.novel_id}, outline exists: {state.outline is not None}")
    
    if state.outline and state.novel_id:
        pool = get_db_pool()
        if pool:
            try:
                async with pool.acquire() as conn:
                    # 先检查是否存在记录，决定插入还是更新
                    await conn.execute("""
                        INSERT INTO novels (novel_id, title, outline, current_volume, current_chapter, current_scene_index, created_at, updated_at)
                        VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW())
                        ON CONFLICT (novel_id) DO UPDATE
                        SET outline = EXCLUDED.outline,
                            current_volume = EXCLUDED.current_volume,
                            current_chapter = EXCLUDED.current_chapter,
                            current_scene_index = EXCLUDED.current_scene_index,
                            updated_at = NOW()
                    """, state.novel_id, 
                        state.outline.get("title", "Untitled"),
                        json.dumps(state.outline),
                        state.current_volume,
                        state.current_chapter,
                        state.current_scene_index if state.current_scene_index is not None else 0)
                logger.info(f"✅ Saved outline for {state.novel_id}")
            except Exception as e:
                logger.error(f"Failed to save outline: {e}", exc_info=True)
    else:
        logger.warning(f"state.outline is empty for {state.novel_id}, cannot save")
    
    # 其他记忆保存（可留空）
    return {}

async def analyze_node(state: AgentState) -> dict[str, Any]:
    intent, subtasks = _keyword_analyze(state.user_input)
    return {"intent": intent, "subtasks": subtasks, "is_complex": _is_complex_task(state.user_input), "current_node": "analyze"}

async def plan_node(state: AgentState) -> dict[str, Any]:
    logger.info(f"plan_node called with chapter={state.current_chapter}, task_type={state.task_type}")
    
    updates = {}
    pool = get_db_pool()
    
    # ===== 如果 outline 丢失，从数据库重新加载 =====
    if state.outline is None and state.novel_id and pool:
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT outline FROM novels WHERE novel_id = $1", state.novel_id
                )
                if row and row["outline"]:
                    state.outline = json.loads(row["outline"])
                    logger.info(f"✅ Reloaded outline for {state.novel_id}")
                    
                    # 计算当前卷的总章节数
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
    
    # ===== 新版plan_node保留新架构上下文注入 =====
    if state.task_type == "scene_plan" and state.novel_id:
        try:
            world = WorldState.from_dict(state.current_state) if state.current_state else WorldState()
            compiler = ContextCompiler()
            # 只传入当前卷的大纲（若有）
            current_volume_outline = None
            if state.outline and "volumes" in state.outline:
                volumes = state.outline.get("volumes", [])
                vol_idx = state.current_volume - 1
                if 0 <= vol_idx < len(volumes):
                    current_volume_outline = volumes[vol_idx]
            compiled = compiler.compile_for_planner(
                world, 
                state.current_volume, 
                state.current_chapter, 
                current_volume_outline or state.outline or {}
            )
            state.metadata["compiled_context"] = compiled
        except Exception as e:
            logger.warning(f"Context compile failed: {e}")
    
    # ===== 调用 PlannerAgent =====
    planner = PlannerAgent()
    planner_updates = await planner.run(state)
    updates.update(planner_updates)
    
    # ===== 处理场景计划（关键修复部分） =====
    if state.task_type == "scene_plan" and updates.get("scene_plan"):
        scene_plan_data = updates["scene_plan"]
        
        # 处理两种可能的格式：直接是数组，或包含 scenes 键的字典
        if isinstance(scene_plan_data, dict):
            scenes = scene_plan_data.get("scenes", [])
        elif isinstance(scene_plan_data, list):
            scenes = scene_plan_data
        else:
            scenes = []
        
        logger.info(f"plan_node: extracted {len(scenes)} scenes from planner response")
        
        if scenes:
            # 确保每个场景都有必须字段
            for i, scene in enumerate(scenes):
                if "must_events" not in scene or not scene["must_events"]:
                    scene["must_events"] = [f"推进主线剧情（场景{i+1}）"]
                if "state_delta" not in scene:
                    scene["state_delta"] = {"events": []}
                if "depends_on" not in scene:
                    scene["depends_on"] = []
                if "scene_id" not in scene:
                    scene["scene_id"] = i + 1
            
            updates["scene_plan_list"] = scenes
            updates["total_scenes_in_chapter"] = len(scenes)
            updates["scene_plan"] = scenes[0] if scenes else {}
            
            # 强制写入 state 以确保 LangGraph 正确合并
            state.scene_plan_list = scenes
            state.total_scenes_in_chapter = len(scenes)
            state.scene_plan = scenes[0] if scenes else {}
            logger.info(f"plan_node: set state.scene_plan (exists={state.scene_plan is not None}, scenes={len(scenes)})")
        else:
            logger.warning("plan_node: no scenes extracted from planner response")
    
    # ===== 确保 total_chapters_in_volume 有值 =====
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
    
    # 调试日志
    logger.info(f"plan_node returning: scene_plan in state = {state.scene_plan is not None}")
    logger.info(f"plan_node returning: scene_plan in updates = {updates.get('scene_plan') is not None}")
    
    return updates

async def writer_node(state: AgentState) -> dict[str, Any]:
    if state.scene_plan is None:
        logger.error("writer_node: state.scene_plan is None, cannot write")
        return {"scene_text": "", "final_answer": "", "current_node": "writer"}
    
    writer = WritingAgent()
    result = await writer.run(state)
    raw_json = result.get("scene_text", "")   # 原始输出（包含 JSON）

    # 提取 JSON 中的 scene_text 作为正文
    import re, json
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
        "scene_text": raw_json,          # 原始 JSON 用于验证器
        "final_answer": clean_text,      # 清洗后的正文用于保存
        "current_node": "writer",
        "deviation_detected": result.get("deviation_detected", False),
        "missing_goal_keywords": result.get("missing_goal_keywords", []),
        "missing_conflict_keywords": result.get("missing_conflict_keywords", []),
    }

async def validate_node(state: AgentState) -> dict[str, Any]:
    if state.scene_plan is None:
        logger.warning("validate_node: state.scene_plan is None, using empty dict")
        state.scene_plan = {}

    mode = "novel" if state.scene_text else "code"
    state.validation_mode = mode
    validator = ValidatorAgent()
    updates = await validator.run(state)

    validation_result = updates.get("validation_result", {})
    passed = validation_result.get("passed", False)
    should_retry = validation_result.get("should_retry", False)
    need_semantic = validation_result.get("need_semantic", False)

    # ========== 异步语义验证（如果需要） ==========
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

    # ========== 处理重试逻辑 ==========
    if not passed and should_retry:
        retry_count = getattr(state, 'retry_count', 0) + 1
        max_retries = getattr(state, 'max_retries_per_subtask', 2)

        logger.warning(f"Validation failed: {validation_result.get('feedback', '')}")

        if retry_count < max_retries:
            logger.info(f"Retrying scene {state.current_scene_index + 1} (retry {retry_count}/{max_retries})")
            return {
                "validation_result": validation_result,
                "error": validation_result.get("feedback"),
                "retry_count": retry_count,
                "needs_retry": True,
            }
        else:
            logger.error(f"Validation failed after {max_retries} retries, skipping scene")
            # 跳过当前场景，继续下一个
            current_idx = state.current_scene_index if state.current_scene_index is not None else 0
            new_idx = current_idx + 1

            base_updates = {
                "validation_result": validation_result,
                "error": validation_result.get("feedback"),
                "current_scene_index": new_idx,
                "retry_count": 0,
                "needs_retry": False,
            }

            # ===== 关键修复：跳过场景后，更新下一个场景的计划 =====
            scene_plan_list = getattr(state, 'scene_plan_list', [])
            if new_idx < len(scene_plan_list):
                next_scene_plan = scene_plan_list[new_idx]
                base_updates["scene_plan"] = next_scene_plan
                logger.info(f"Skipped scene, updated scene_plan for next scene (index {new_idx})")
            else:
                # 本章所有场景已处理完，稍后会由章节完成逻辑清空
                pass

            # 检查是否完成了本章所有场景（包括跳过的）
            total_scenes = getattr(state, "total_scenes_in_chapter", 0)
            if total_scenes > 0 and new_idx >= total_scenes:
                logger.info(f"Chapter {state.current_chapter} completed (with skipped scenes), advancing to next chapter")
                base_updates["current_chapter"] = state.current_chapter + 1
                base_updates["current_scene_index"] = 0
                base_updates["scene_plan_list"] = []
                base_updates["total_scenes_in_chapter"] = 0
                base_updates["_chapter_finished"] = True
                base_updates["scene_plan"] = None   # <-- 添加

                # 检查是否需要切换到下一卷
                total_chapters = getattr(state, "total_chapters_in_volume", 0)
                if total_chapters > 0 and (state.current_chapter + 1) > total_chapters:
                    base_updates["current_volume"] = state.current_volume + 1
                    base_updates["current_chapter"] = 1
                    logger.info(f"Volume {state.current_volume} completed! Moving to volume {state.current_volume + 1}")

            return base_updates

    # ========== 验证失败但不重试（致命错误） ==========
    if not passed:
        logger.error(f"Validation failed (fatal): {validation_result.get('feedback', '')}")
        return {
            "validation_result": validation_result,
            "error": validation_result.get("feedback"),
            "needs_retry": False,
        }

    # ========== 验证通过后的处理 ==========
    # 如果有小说场景，应用 planned_delta 并存储事件/快照
    if state.task_type == "scene_plan" and state.scene_text:
        pool = get_db_pool()
        if pool:
            try:
                current_world = WorldState.from_dict(state.current_state) if state.current_state else WorldState()
                planned_delta_dict = state.scene_plan.get("state_delta", {}) if state.scene_plan else {}
                if planned_delta_dict:
                    delta = StateDelta.from_dict(planned_delta_dict)
                    new_world = delta.apply_to(current_world)
                    # 存储事件
                    event_store = NarrativeEventStore(pool)
                    for evt in delta.events:
                        await event_store.append_event(state.novel_id, evt, state.current_volume, state.current_chapter)
                    last_id = await event_store.get_last_event_id(state.novel_id)
                    snap_mgr = SnapshotManager(pool)
                    total_scenes = getattr(state, "total_scenes_in_chapter", 0)
                    chapter_finished = total_scenes > 0 and (state.current_scene_index + 1 >= total_scenes)
                    if chapter_finished:
                        await snap_mgr.save_snapshot(state.novel_id, new_world, last_id, state.current_volume, state.current_chapter)
                    updates["current_state"] = new_world.to_dict()
            except Exception as e:
                logger.error(f"Delta apply error: {e}")

        # 更新场景索引
        current_idx = state.current_scene_index if state.current_scene_index is not None else 0
        new_idx = current_idx + 1
        updates["current_scene_index"] = new_idx
        updates["retry_count"] = 0

        # ===== 关键修复：更新下一个场景的计划 =====
        scene_plan_list = getattr(state, 'scene_plan_list', [])
        if new_idx < len(scene_plan_list):
            next_scene_plan = scene_plan_list[new_idx]
            updates["scene_plan"] = next_scene_plan
            logger.info(f"Updated scene_plan for next scene (index {new_idx}): goal={next_scene_plan.get('goal', '')[:60]}")
        else:
            # 本章所有场景已完成，稍后清空
            pass

        # 保存正文到文件
        if state.novel_id and state.final_answer:
            await _save_scene_to_file(state, state.final_answer)

        # 处理章节完成
        total_scenes = getattr(state, "total_scenes_in_chapter", 0)
        logger.info(f"validate_node: total_scenes={total_scenes}, current_idx={current_idx}, new_idx={new_idx}, current_chapter={state.current_chapter}")

        if total_scenes > 0 and new_idx >= total_scenes:
            new_chapter = state.current_chapter + 1
            logger.info(f"✅ Chapter {state.current_chapter} completed! Advancing to chapter {new_chapter}")
            updates["current_chapter"] = new_chapter
            updates["current_scene_index"] = 0
            updates["scene_plan_list"] = []
            updates["total_scenes_in_chapter"] = 0
            updates["_chapter_finished"] = True
            updates["scene_plan"] = None   # <-- 添加

            total_chapters = getattr(state, "total_chapters_in_volume", 0)
            if total_chapters > 0 and new_chapter > total_chapters:
                new_vol = state.current_volume + 1
                updates["current_volume"] = new_vol
                updates["current_chapter"] = 1
                logger.info(f"📚 Volume {state.current_volume} completed! Moving to volume {new_vol}")
        else:
            logger.info(f"Continuing with next scene of chapter {state.current_chapter}")

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