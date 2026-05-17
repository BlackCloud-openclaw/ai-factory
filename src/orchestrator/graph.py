import uuid
from typing import Any, Dict

from langgraph.graph import StateGraph, END

from src.orchestrator.state import AgentState
from src.orchestrator.nodes import (
    analyze_node,
    research_node,
    code_node,
    validate_node,
    load_memory_node,
    save_memory_node,
    scheduler_node,
    advance_subtask_node,
    tool_node_v2,
    writer_node, 
    plan_node,
)
from src.agents.planner import PlannerAgent
from src.scheduler.task_scheduler import TaskScheduler
from src.common.logging import setup_logging

logger = setup_logging("orchestrator.graph")


def route_after_analyze(state: AgentState) -> str:
    # 续写模式：直接进入写作节点，不重新规划，避免重置进度
    if state.resume:
        return "writer"

    task_type = getattr(state, 'task_type', 'code')
    if task_type == "scene_plan" and state.scene_plan:
        return "writer"
    if task_type == "novel_outline" and state.outline:
        return "save_memory"
    return "planning"


def after_plan(state: AgentState) -> str:
    if state.pending_tool_calls:
        return "tool_node"
    
    task_type = getattr(state, 'task_type', 'code')
    
    if task_type == "novel_outline":
        return "save_memory"
    
    if task_type == "scene_plan" and state.scene_plan:
        return "writer"
    
    task_plan = getattr(state, 'task_plan', None)
    if not task_plan:
        return 'save_memory'
    subtasks = task_plan.get('subtasks', [])
    if len(subtasks) > 1:
        return 'scheduler'
    elif len(subtasks) == 1:
        return 'code'
    return 'save_memory'


def route_after_scheduler(state: AgentState) -> str:
    return 'save_memory'

def route_after_research(state: AgentState) -> str:
    subtasks = state.subtasks if hasattr(state, 'subtasks') else state.get('subtasks', [])
    return 'code' if subtasks else END


def route_after_code(state: AgentState) -> str:
    exec_result = state.execution_result if hasattr(state, 'execution_result') else state.get('execution_result', {})
    if exec_result and not exec_result.get('success', False):
        needs_retry = state.should_retry()
        if needs_retry:
            return 'code'
    return 'validate'


def route_after_validate(state: AgentState) -> str:   
    if state.task_type == "scene_plan":
        total_scenes = getattr(state, 'total_scenes_in_chapter', 0)
        current_scene_index = getattr(state, 'current_scene_index', 0)
        validation_result = state.validation_result or {}
        passed = validation_result.get("passed", False)
        should_retry = validation_result.get("should_retry", False)
        retry_count = getattr(state, 'retry_count', 0)
        max_retries = getattr(state, 'max_retries_per_subtask', 2)
        current_chapter = state.current_chapter
        total_chapters = getattr(state, 'total_chapters_in_volume', 0)

        logger.info(f"route_after_validate: total_scenes={total_scenes}, current_scene_index={current_scene_index}, passed={passed}")

        if not passed and should_retry and retry_count < max_retries:
            new_retry_count = retry_count + 1
            logger.info(f"Scene validation failed, retrying ({new_retry_count}/{max_retries}): {validation_result.get('feedback', '')[:100]}")
            return "writer"

        if not passed and retry_count >= max_retries:
            logger.warning(f"Scene validation failed after {max_retries} retries, skipping scene")
            return "validate"

        if passed:
            logger.info("Scene validation passed")
            if total_scenes > 0 and current_scene_index < total_scenes:
                return "writer"
            if total_chapters > 0 and current_chapter <= total_chapters:
                logger.info(f"Chapter finished, moving to planning for chapter {current_chapter} of {total_chapters}")
                return "planning"
            else:
                logger.info("All chapters completed, ending.")
                return "save_memory"

        logger.warning(f"Validation failed without retry: {validation_result.get('feedback', '')}")
        return "save_memory"    

    # 原有的代码验证逻辑（保持不变）
    validation_result = state.validation_result or {}
    passed = validation_result.get("passed", False)
    retry_count = getattr(state, 'retry_count', 0)
    max_retries = getattr(state, 'max_retries_per_subtask', 2)
    remaining = getattr(state, 'remaining_subtasks', [])

    if not passed and retry_count < max_retries:
        return "code"
    if passed:
        return "advance_subtask" if remaining else "save_memory"
    return "research" if retry_count < max_retries else "save_memory"

def create_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("load_memory", load_memory_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("planning", plan_node)
    workflow.add_node("scheduler", scheduler_node)
    workflow.add_node("research", research_node)
    workflow.add_node("code", code_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("save_memory", save_memory_node)
    workflow.add_node("advance_subtask", advance_subtask_node)
    workflow.add_node("tool_node", tool_node_v2)
    workflow.add_node("writer", writer_node)

    workflow.set_entry_point("load_memory")
    workflow.add_edge("load_memory", "analyze")

    workflow.add_conditional_edges(
        "analyze",
        route_after_analyze,
        {
            "planning": "planning",
            "code": "code",
            "research": "research",
            "writer": "writer",
            "save_memory": "save_memory",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "planning",
        after_plan,
        {
            "scheduler": "scheduler",
            "code": "code",
            "save_memory": "save_memory",
            "research": "research",
            "tool_node": "tool_node",
            "writer": "writer",
        },
    )

    workflow.add_conditional_edges(
        "scheduler",
        route_after_scheduler,
        {
            "validate": "validate",
            "save_memory": "save_memory",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "research",
        route_after_research,
        {
            "code": "code",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "code",
        route_after_code,
        {
            "code": "code",
            "validate": "validate",
        },
    )

    workflow.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "writer": "writer",
            "planning": "planning",
            "save_memory": "save_memory",
            "research": "research",
            "advance_subtask": "advance_subtask",
            "code": "code",
        },
    )

    workflow.add_edge("save_memory", END)
    workflow.add_edge("advance_subtask", "code")
    workflow.add_edge("tool_node", "planning")
    workflow.add_edge("writer", "validate")

    return workflow


def compile_workflow() -> any:
    workflow = create_workflow()
    return workflow.compile()