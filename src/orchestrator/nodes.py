# src/orchestrator/nodes.py
import uuid
import time
import json
import re
from typing import Any
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.orchestrator.state import AgentState
from src.agents.research import ResearchAgent
from src.agents.executor import ExecutorAgent
from src.agents.memory import MemoryAgent
from src.agents.planner import PlannerAgent
from src.agents.validator import ValidatorAgent
from src.common.logging import setup_logging
from src.writing.event_store import EventStore
from src.writing.reducer import apply_event
from src.writing.invariants import validate_event
from src.agents.writer import WritingAgent
from src.writing.events import Event
from src.execution.tools_registry import ToolsRegistry
from src.db import get_db_pool

logger = setup_logging("orchestrator.nodes")

# Global memory agent instance (shared across requests)
_memory_agent = MemoryAgent()


def get_memory_agent() -> MemoryAgent:
    return _memory_agent


# ============================================================================
# Helper functions
# ============================================================================

def _keyword_analyze(user_input: str) -> tuple[str, list[str]]:
    lower = user_input.lower()
    if any(kw in lower for kw in ["write", "code", "implement", "function", "class", "create"]):
        intent = "code_generation"
    elif any(kw in lower for kw in ["explain", "what is", "how does", "tell me", "research", "knowledge"]):
        intent = "research"
    else:
        intent = "general_chat"
    subtasks = [user_input]
    return intent, subtasks


COMPLEXITY_KEYWORDS = [
    "并且", "然后", "先", "再", "接着", "最后", "同时", "还要", "另外",
    "and", "then", "also", "finally", "next", "after", "before", "while",
    "multiple", "sequence", "pipeline", "workflow"
]

def _is_complex_task(user_input: str) -> bool:
    if len(user_input) > 200:
        return True
    lower = user_input.lower()
    return any(kw in lower for kw in COMPLEXITY_KEYWORDS)


def _build_research_summary(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No research results available."
    summaries = []
    for r in results:
        summary = r.get("summary", r.get("content", "No content"))
        source = r.get("source", "unknown")
        summaries.append(f"[{source}]: {summary}")
    return "\n\n".join(summaries)


def _heuristic_validate(state: AgentState) -> bool:
    if state.code_generated and len(state.code_generated.strip()) > 10:
        if state.execution_result:
            return state.execution_result.get("success", False)
        return True
    if state.research_results:
        return True
    return False


# ============================================================================
# Node functions
# ============================================================================

async def load_memory_node(state: AgentState) -> dict[str, Any]:
    return await _memory_agent.run(state)


async def save_memory_node(state: AgentState) -> dict[str, Any]:
    """Save important context after workflow completion."""
    project_id = state.project_id or state.metadata.get("session_id", "default")
    memory_agent = get_memory_agent()
    logger.info(f"Saving memory for project={project_id}")

    try:
        await memory_agent.store(
            project_id=project_id,
            key="last_intent",
            value=state.intent,
            metadata={"timestamp": time.time()},
        )
        if state.subtasks:
            await memory_agent.store(
                project_id=project_id,
                key="last_subtasks",
                value=state.subtasks,
                metadata={"timestamp": time.time()},
            )
        if state.code_generated:
            await memory_agent.store(
                project_id=project_id,
                key="last_code",
                value=state.code_generated,
                metadata={"timestamp": time.time(), "file_path": state.code_file_path},
            )
        if state.execution_result:
            exec_summary = {
                "success": state.execution_result.get("success", False),
                "stdout": str(state.execution_result.get("stdout", ""))[:200],
                "stderr": str(state.execution_result.get("stderr", ""))[:200],
            }
            await memory_agent.store(
                project_id=project_id,
                key="last_execution",
                value=exec_summary,
                metadata={"timestamp": time.time()},
            )
        conversation_entry = {
            "user_input": state.user_input,
            "intent": state.intent,
            "code_generated": bool(state.code_generated),
            "execution_success": (
                state.execution_result.get("success", False) if state.execution_result else None
            ),
            "timestamp": time.time(),
        }
        await memory_agent.append_to_memory(
            project_id=project_id,
            key="conversation_history",
            value=conversation_entry,
            max_items=50,
        )
        if state.final_answer:
            await memory_agent.store(
                project_id=project_id,
                key="last_answer",
                value=state.final_answer[:1000],
                metadata={"timestamp": time.time()},
            )
        logger.info(f"Saved memory for project={project_id}: intent={state.intent}, code={'yes' if state.code_generated else 'no'}")
    except Exception as e:
        logger.error(f"Failed to save memory for project={project_id}: {e}", exc_info=True)

    # 保存小说大纲到 novels 表（如果存在）
    if state.outline and state.novel_id:
        pool = get_db_pool()
        if pool:
            import json
            try:
                async with pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO novels (novel_id, title, outline, current_volume, current_chapter, current_scene, current_state, last_sequence_id, created_at, updated_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW(), NOW())
                        ON CONFLICT (novel_id) DO UPDATE
                        SET title = EXCLUDED.title,
                            outline = EXCLUDED.outline,
                            current_volume = EXCLUDED.current_volume,
                            current_chapter = EXCLUDED.current_chapter,
                            current_scene = EXCLUDED.current_scene,
                            current_state = EXCLUDED.current_state,
                            last_sequence_id = EXCLUDED.last_sequence_id,
                            updated_at = NOW()
                    """, state.novel_id, 
                        state.outline.get("title", "Untitled"),
                        json.dumps(state.outline),
                        state.current_volume,
                        state.current_chapter,
                        state.current_scene,
                        json.dumps({}),  # 初始空状态
                        0)
                logger.info(f"Saved novel outline for {state.novel_id}")
            except Exception as e:
                logger.error(f"Failed to save novel outline: {e}")
                
     # ---- 保存小说正文到文件系统 ----
    if state.scene_text and state.novel_id:
        raw_text = state.scene_text   # 已经清洗
        pool = get_db_pool()
        if pool:
            try:
                novel_data_dir = Path(f"data/novels/{state.novel_id}")
                volumes_dir = novel_data_dir / f"vol_{state.current_volume:03d}"
                volumes_dir.mkdir(parents=True, exist_ok=True)
                chapter_file = volumes_dir / f"chap_{state.current_chapter:03d}.txt"

                # 决定写入模式：如果文件存在，则在写入新场景前先添加分隔符
                if chapter_file.exists():
                    with open(chapter_file, "a", encoding="utf-8") as f:
                        f.write("\n\n<!-- scene break -->\n\n")
                        f.write(raw_text)
                else:
                    with open(chapter_file, "w", encoding="utf-8") as f:
                        f.write(raw_text)

                word_count = len(raw_text)
                # ... 后续数据库更新逻辑（注意：累加字数时需加上之前的字数）
                # 建议：每次更新 chapters 表的 word_count 时，累加而不是覆盖
                async with pool.acquire() as conn:
                    existing = await conn.fetchrow(
                        "SELECT word_count FROM chapters WHERE novel_id=$1 AND volume_num=$2 AND chapter_num=$3",
                        state.novel_id, state.current_volume, state.current_chapter
                    )
                    if existing:
                        new_word_count = existing["word_count"] + word_count
                        await conn.execute("""
                            UPDATE chapters
                            SET word_count = $1, file_path = $2, updated_at = NOW()
                            WHERE novel_id=$3 AND volume_num=$4 AND chapter_num=$5
                        """, new_word_count, str(chapter_file), state.novel_id, state.current_volume, state.current_chapter)
                    else:
                        chapter_id = f"{state.novel_id}_v{state.current_volume}_c{state.current_chapter}"
                        await conn.execute("""
                            INSERT INTO chapters (chapter_id, novel_id, volume_num, chapter_num, file_path, word_count, created_at)
                            VALUES ($1, $2, $3, $4, $5, $6, NOW())
                        """, chapter_id, state.novel_id, state.current_volume, state.current_chapter, str(chapter_file), word_count)

                    # 更新主表进度：注意使用新的 current_scene_index（已完成的数量）
                    await conn.execute("""
                        UPDATE novels
                        SET current_volume = $1, current_chapter = $2, current_scene = $3, updated_at = NOW()
                        WHERE novel_id = $4
                    """, state.current_volume, state.current_chapter, state.current_scene_index, state.novel_id)

                logger.info(f"Saved scene to {chapter_file}, added {word_count} chars")
            except Exception as e:
                logger.error(f"Failed to save novel scene: {e}", exc_info=True)

    return {}    

async def analyze_node(state: AgentState) -> dict[str, Any]:
    logger.info(f"Analyzing user input: {state.user_input[:200]}...")
    intent, subtasks = _keyword_analyze(state.user_input)
    is_complex = _is_complex_task(state.user_input)
    logger.info(f"Intent: {intent}, Subtasks: {subtasks}, is_complex: {is_complex}")
    return {
        "intent": intent,
        "subtasks": subtasks,
        "is_complex": is_complex,
        "current_node": "analyze",
    }

async def plan_node(state: AgentState) -> dict[str, Any]:
    """生成任务计划，并针对 scene_plan 模式设置场景列表和总数"""
    planner = PlannerAgent()
    updates = await planner.run(state)
    if "error" not in updates:
        updates["error"] = None

    # 处理小说场景计划模式
    if state.task_type == "scene_plan" and updates.get("scene_plan"):
        scene_plan_data = updates["scene_plan"]
        # 期望格式为 {"scenes": [...]} 或直接是列表
        if isinstance(scene_plan_data, dict) and "scenes" in scene_plan_data:
            scenes = scene_plan_data["scenes"]
        elif isinstance(scene_plan_data, list):
            scenes = scene_plan_data
        else:
            # 降级处理：单个场景计划
            scenes = [scene_plan_data]

        if scenes:
            updates["scene_plan_list"] = scenes
            updates["total_scenes_in_chapter"] = len(scenes)
            # 取第一个场景计划作为当前要生成的场景
            updates["scene_plan"] = scenes[0]
            updates["current_scene_index"] = 0
            logger.info(f"plan_node: total_scenes_in_chapter={len(scenes)}, scene_plan_list length={len(scenes)}")
        else:
            logger.error("plan_node: No scenes extracted from scene_plan_data")
            updates["error"] = "No scenes generated"

    return updates   

async def scheduler_node(state: AgentState) -> dict[str, Any]:
    plan_data = getattr(state, 'task_plan', None)
    if not plan_data:
        plan_data = getattr(state, 'plan', None)
    if not plan_data:
        logger.warning("No plan data found in state for scheduler node")
        return {"plan_status": "no_plan", "subtask_results": {}, "current_node": "scheduler"}

    from src.scheduler.task_scheduler import TaskScheduler
    from src.agents.planner import TaskPlan, Subtask

    subtasks_data = plan_data.get("subtasks", [])
    if not subtasks_data:
        logger.warning("No subtasks in plan, skipping scheduler")
        return {"plan_status": "no_subtasks", "subtask_results": {}, "current_node": "scheduler"}

    subtasks = []
    for sd in subtasks_data:
        subtasks.append(
            Subtask(
                id=sd.get("id", f"st_{len(subtasks):03d}"),
                name=sd.get("name", sd.get("description", "")),
                description=sd.get("description", ""),
                type=sd.get("type", "code"),
                dependencies=sd.get("dependencies", []),
                required_tools=sd.get("required_tools", []),
            )
        )

    task_plan = TaskPlan(
        plan_id=plan_data.get("plan_id", f"plan_{uuid.uuid4().hex[:8]}"),
        original_request=state.user_input,
        subtasks=subtasks,
    )

    scheduler = TaskScheduler(max_concurrent=3, max_retries=2)
    task_id = await scheduler.submit_plan(task_plan)
    summary = await scheduler.run(task_id)

    subtask_results = summary.get("results", {})
    success_count = summary.get("success", 0)
    fail_count = summary.get("failed", 0)

    code_outputs = []
    research_outputs = []
    for st_id, result in subtask_results.items():
        if result.get("status") == "success":
            raw = result.get("result", {})
            if raw.get("type") == "code":
                code_outputs.append(raw.get("output", ""))
            elif raw.get("type") == "research":
                research_outputs.append(raw.get("output", ""))

    merged_code = "\n\n".join(code_outputs) if code_outputs else ""
    merged_research = "\n\n".join(research_outputs) if research_outputs else ""

    plan_status = "success" if fail_count == 0 else "partial"
    if success_count == 0:
        plan_status = "failed"

    logger.info(f"Scheduler completed: {success_count} success, {fail_count} failed, status={plan_status}")

    return {
        "task_id": task_id,
        "subtask_results": subtask_results,
        "plan_status": plan_status,
        "code_generated": merged_code,
        "research_results": ([{"summary": r, "source": "scheduler"} for r in research_outputs] if research_outputs else state.research_results),
        "execution_result": ({"success": True, "stdout": merged_code[:500]} if merged_code and success_count > 0 else None),
        "current_node": "scheduler",
    }


async def research_node(state: AgentState) -> dict[str, Any]:
    logger.info(f"Running research for: {state.user_input[:200]}...")
    research_agent = ResearchAgent()
    result = await research_agent.run(state)
    return {
        "research_results": result.get("research_results", []),
        "sources": result.get("sources", []),
        "current_node": "research",
    }


async def code_node(state: AgentState) -> dict[str, Any]:
    logger.info(f"Running code generation for subtasks: {state.subtasks}")
    executor = ExecutorAgent()
    updates = await executor.run(state)
    research_results = state.research_results
    return {
        "code_generated": updates.get("code_generated", ""),
        "code_file_path": updates.get("code_file_path", ""),
        "execution_result": updates.get("execution_result"),
        "research_results": research_results,
        "current_node": "code",
    }


async def validate_node(state: AgentState) -> dict[str, Any]:
    logger.info("Running validation")

    # 根据是否存在场景正文切换验证模式
    if state.scene_text:
        state.validation_mode = "novel"
        logger.info("Switching validation mode to 'novel' due to scene_text present")
    else:
        state.validation_mode = "code"

    validator = ValidatorAgent()
    updates = await validator.run(state)
    validation_result = updates.get("validation_result", {})
    passed = validation_result.get("passed", False)
    feedback = validation_result.get("feedback", "")

    logger.info(f"Validation result: passed={passed}, feedback={feedback}")

    retry_count = getattr(state, 'retry_count', 0)
    if not passed:
        retry_count += 1

    max_retries = getattr(state, 'max_retries_per_subtask', 2)
    needs_retry = retry_count < max_retries

    final_answer = updates.get("final_answer", "")
    if not final_answer and passed:
        final_answer = state.code_generated or _build_research_summary(state.research_results)

    if passed:
        updates_dict = {
            "validation_result": validation_result,
            "final_answer": final_answer,
            "current_node": "validate",
            "error": None,
            "needs_retry": False,
            "retry_count": retry_count,
        }

        # 小说场景计划模式下 – 处理场景索引递增和章节切换
        if state.task_type == "scene_plan":
            new_index = state.current_scene_index + 1
            updates_dict["current_scene_index"] = new_index
            total_scenes = getattr(state, "total_scenes_in_chapter", 0)
            logger.info(f"validate_node: current_scene_index BEFORE={state.current_scene_index}, new_index={new_index}, total_scenes={total_scenes}")

            if total_scenes > 0 and new_index >= total_scenes:
                # 本章所有场景已完成，切换到下一章
                logger.info("Chapter completed! Advancing to next chapter.")
                updates_dict["current_chapter"] = state.current_chapter + 1
                updates_dict["current_scene_index"] = 0          # 重置索引
                updates_dict["scene_plan_list"] = []            # 清空计划，触发重新生成
                updates_dict["_chapter_finished"] = True   # 新增标志
                # 可选：卷切换逻辑（如果 total_chapters_in_volume 已设置）
                total_chapters = getattr(state, "total_chapters_in_volume", 0)
                if total_chapters > 0 and updates_dict["current_chapter"] > total_chapters:
                    updates_dict["current_volume"] = state.current_volume + 1
                    updates_dict["current_chapter"] = 1
                    logger.info(f"Volume completed! New volume {updates_dict['current_volume']}")
            else:
                # 本章尚未完成，保留现有计划列表，只更新索引
                logger.info(f"Chapter not yet complete. Next scene index {new_index} of {total_scenes}")

        return updates_dict
    else:
        # 验证失败，重试或结束
        return {
            "validation_result": validation_result,
            "error": feedback,
            "retry_count": retry_count,
            "current_node": "validate",
            "needs_retry": needs_retry,
        } 

def advance_subtask_node(state: AgentState) -> dict[str, Any]:
    remaining = getattr(state, 'remaining_subtasks', []) or []
    if remaining:
        next_task = remaining[0]
        new_remaining = remaining[1:]
        current_index = getattr(state, 'current_subtask_index', 0) or 0
        return {
            "subtasks": [next_task["description"]],
            "current_subtask_index": current_index + 1,
            "current_subtask_id": next_task["id"],
            "remaining_subtasks": new_remaining,
            "validation_result": None,
            "execution_result": None,
            "needs_retry": False,
            "retry_count": 0,
        }
    return {"subtasks": []}

async def writer_node(state: AgentState) -> dict[str, Any]:
    updates = {}
    # 1. 如果当前状态为空，尝试从数据库加载快照
    if not state.current_state and state.novel_id:
        pool = get_db_pool()
        if pool:
            store = EventStore(pool)
            snap_state, snap_seq = await store.load_snapshot(state.novel_id)
            if snap_state:
                events = await store.load_events(state.novel_id, from_sequence=snap_seq)
                current = snap_state
                for evt in events:
                    current = apply_event(current, evt)
                updates["current_state"] = current
                updates["last_sequence_id"] = events[-1].sequence_id if events else snap_seq

    # 2. 准备当前场景计划（支持批量计划列表）
    if state.scene_plan_list and state.current_scene_index < len(state.scene_plan_list):
        # 从列表取出当前索引的场景计划
        current_scene_plan = state.scene_plan_list[state.current_scene_index]
    else:
        # 降级：使用单场景计划
        current_scene_plan = state.scene_plan

    # 临时覆盖 state.scene_plan，供 WritingAgent 使用
    state.scene_plan = current_scene_plan

    # 3. 调用 WritingAgent 生成原始场景文本
    writer = WritingAgent()
    result = await writer.run(state)
    raw_text = result.get("scene_text", "")

    # ========== 4. 清洗正文：移除所有思考过程、计划、标记 ==========
    clean_text = raw_text
    if raw_text:
        # 优先查找 (Start Writing) 标记
        if '(Start Writing)' in raw_text:
            clean_text = raw_text.split('(Start Writing)', 1)[1].strip()
        else:
            # 查找第一个中文字符（正文必然包含汉字）
            match = re.search(r'([\u4e00-\u9fff])', raw_text)
            if match:
                start = match.start()
                # 可选：向前找到最近的换行，避免从单词中间截断
                line_start = raw_text.rfind('\n', 0, start) + 1
                clean_text = raw_text[line_start:].strip()
            else:
                clean_text = raw_text

        # 删除残留的剧本标记（如 "*Twist/Climax:*"、"*Resolution:*" 等）
        clean_text = re.sub(r'\n*\*{1,2}[^*]+\*{1,2}[:：]?\s*\n', '\n', clean_text)
        # 删除以 "*(Check" 开头的行及其之后所有内容
        clean_text = re.sub(r'\n?\s*\*\(Check.*$', '', clean_text, flags=re.DOTALL)
        # 删除孤立的英文注释行（如 "Expansion Plan:"）
        clean_text = re.sub(r'\n\s*[A-Za-z].*?:\s*(?:.*\n)*?', '\n', clean_text)
        # 删除以数字加点开头的英文行（例如 "1.  Task: Write..."）
        clean_text = re.sub(r'^\s*\d+\.\s+[A-Za-z].*\n', '', clean_text, flags=re.MULTILINE)
        # 删除以破折号开头的英文行
        clean_text = re.sub(r'^\s*-\s+[A-Za-z].*\n', '', clean_text, flags=re.MULTILINE)
        # 压缩多余空行
        clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text).strip()
    # ==================================================

    # 5. 返回更新字段
    return {
        **updates,
        "scene_text": clean_text,
        "final_answer": clean_text,
        "current_node": "writer",
    }

async def tool_node_v2(state: AgentState) -> dict[str, Any]:
    logger.info("Entering tool_node_v2")

    pool = get_db_pool()
    if pool is None:
        raise RuntimeError("Database pool not initialized")

    store = EventStore(pool)
    registry = ToolsRegistry()

    if not state.novel_id:
        logger.warning("No novel_id in state, falling back to simple tool execution")
        return await _tool_node_fallback(state)
    novel_id = state.novel_id

    current_state = state.current_state
    last_sequence_id = state.last_sequence_id
    if not current_state:
        snapshot_state, snapshot_seq = await store.load_snapshot(novel_id)
        if snapshot_state:
            current_state = snapshot_state
            last_sequence_id = snapshot_seq
        else:
            current_state = {}
            last_sequence_id = 0
        events = await store.load_events(novel_id, from_sequence=last_sequence_id)
        for evt in events:
            current_state = apply_event(current_state, evt)
            last_sequence_id = evt.sequence_id

    events_applied = []
    updated_state = current_state.copy()

    for call in state.pending_tool_calls:
        tool_name = call.get("tool")
        args = call.get("args", {})
        tool_func = registry.get_tool_function(tool_name)
        if not tool_func:
            logger.error(f"Tool '{tool_name}' not found in registry")
            continue

        try:
            result = await tool_func(**args)
        except Exception as e:
            logger.error(f"Tool {tool_name} execution failed: {e}")
            continue

        if isinstance(result, Event):
            event = result
        elif isinstance(result, dict) and "type" in result:
            event = Event.new(
                event_type=result["type"],
                payload=result.get("payload", {}),
                novel_id=novel_id,
                chapter_id=state.chapter_id
            )
        else:
            logger.warning(f"Tool {tool_name} did not return a valid event, skipping")
            continue

        ok, msg = validate_event(updated_state, event)
        if not ok:
            logger.warning(f"Event validation failed: {msg}")
            continue

        seq = await store.insert_event(event)
        event.sequence_id = seq
        updated_state = apply_event(updated_state, event)
        events_applied.append(event)

    total_events_after = last_sequence_id + len(events_applied)
    if total_events_after // 100 > (state.last_sequence_id // 100):
        await store.save_snapshot(novel_id, updated_state, total_events_after)
        if events_applied:
            logger.info(f"Saved snapshot for novel {novel_id} at sequence {total_events_after}")

    return {
        "pending_tool_calls": [],
        "applied_events": state.applied_events + events_applied,
        "current_state": updated_state,
        "last_sequence_id": events_applied[-1].sequence_id if events_applied else state.last_sequence_id,
        "current_node": "tool_node",
    }


async def _tool_node_fallback(state: AgentState) -> dict[str, Any]:
    registry = ToolsRegistry()
    results = []
    for call in state.pending_tool_calls:
        tool_name = call.get("tool")
        args = call.get("args", {})
        tool_func = registry.get_tool_function(tool_name)
        if tool_func:
            try:
                res = await tool_func(**args)
                results.append(res)
            except Exception as e:
                logger.error(f"Tool {tool_name} failed: {e}")
    return {
        "pending_tool_calls": [],
        "tool_results": results,
        "current_node": "tool_node",
    }