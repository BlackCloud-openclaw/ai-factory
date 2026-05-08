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
    tool_node_v2,   # 新增导入
    writer_node,          # 新增
)
from src.agents.planner import PlannerAgent
from src.scheduler.task_scheduler import TaskScheduler
from src.common.logging import setup_logging

logger = setup_logging("orchestrator.graph")


def route_after_analyze(state: AgentState) -> str:
    """Decide which node to go to after analysis.

    Always routes to planning for complex tasks or when subtasks are detected.
    For simple tasks with no subtasks, routes to planning which may
    create a single-subtask plan and then route accordingly.
    """
    # 如果已经有场景计划，直接进入写作节点，跳过 planning
    task_type = getattr(state, 'task_type', 'code')
    if task_type == "scene_plan" and state.scene_plan:
        return "writer"
    if task_type == "novel_outline" and state.outline:
        return "save_memory"
    return "planning"


def after_plan(state: AgentState) -> str:
    """
    决定规划后进入哪个节点。
    优先处理工具调用，否则根据任务计划路由。
    """
    # 如果有待处理的工具调用，进入 tool_node
    if state.pending_tool_calls:
        return "tool_node"
    
    task_type = getattr(state, 'task_type', 'code')
    
    # 小说大纲生成后直接保存，不进入 code/scheduler
    if task_type == "novel_outline":
        return "save_memory"
    
    # 场景计划生成后进入写作节点
    if task_type == "scene_plan" and state.scene_plan:
        return "writer"
    
    # 原有代码生成流程
    task_plan = getattr(state, 'task_plan', None)
    if not task_plan:
        return 'save_memory'
    subtasks = task_plan.get('subtasks', [])
    if len(subtasks) > 1:
        return 'scheduler'
    elif len(subtasks) == 1:
        # 可以进一步根据类型路由，但简单返回 code 通常可行
        return 'code'
    return 'save_memory'


def route_after_scheduler(state: AgentState) -> str:
    """Decide which node to go to after scheduler completes.

    Routes to validate if there's code output, otherwise to save_memory.
    """
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
        validation_result = state.validation_result or {}
        passed = validation_result.get("passed", False)
        retry_count = getattr(state, 'retry_count', 0)
        max_retries = getattr(state, 'max_retries_per_subtask', 2)

        if not passed and retry_count < max_retries:
            return "writer"
        elif passed:
            # 如果场景计划列表为空且总场景数 > 0，说明本章所有场景已生成完毕，需要生成下一章的计划
            if not state.scene_plan_list and getattr(state, "total_scenes_in_chapter", 0) > 0:
                total_chapters = getattr(state, "total_chapters_in_volume", 0)
                if state.current_chapter < total_chapters:
                    logger.info("Chapter finished, moving to planning for next chapter.")
                    return "planning"
                else:
                    logger.info("All chapters completed, ending.")
                    return "save_memory"
            # 正常情况：还有场景未生成
            total_scenes = getattr(state, "total_scenes_in_chapter", 0)
            if total_scenes > 0 and state.current_scene_index < total_scenes:
                return "writer"
            else:
                # 没有剩余场景且 scene_plan_list 非空（理论上不会走到这里）兜底
                return "save_memory"
        else:
            return "save_memory"

    # 原有代码模式逻辑（保持不变）
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


async def plan_node(state: AgentState) -> dict[str, Any]:
    planner = PlannerAgent()
    updates = await planner.run(state)
    if "error" not in updates:
        updates["error"] = None

    # 如果是 scene_plan 且生成了场景列表，设置总场景数
    if state.task_type == "scene_plan" and updates.get("scene_plan"):
        scene_plan_data = updates["scene_plan"]
        if isinstance(scene_plan_data, dict) and "scenes" in scene_plan_data:
            scenes = scene_plan_data["scenes"]
            updates["scene_plan_list"] = scenes
            updates["total_scenes_in_chapter"] = len(scenes)
            if scenes:
                updates["scene_plan"] = scenes[0]
                updates["current_scene_index"] = 0
    return updates


def create_workflow() -> StateGraph:
    """Build the LangGraph workflow for AI Factory."""

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("load_memory", load_memory_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("planning", plan_node)
    workflow.add_node("scheduler", scheduler_node)
    workflow.add_node("research", research_node)
    workflow.add_node("code", code_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("save_memory", save_memory_node)
    workflow.add_node("advance_subtask", advance_subtask_node)
    workflow.add_node("tool_node", tool_node_v2)   # 新增 tool_node
    workflow.add_node("writer", writer_node)

    # Set entry point
    workflow.set_entry_point("load_memory")

    # load_memory -> analyze (always)
    workflow.add_edge("load_memory", "analyze")

    # Edges from analyze (conditional)
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

    # Edges from planning (conditional) - 修改为支持 tool_node
    workflow.add_conditional_edges(
        "planning",
        after_plan,
        {
            "scheduler": "scheduler",
            "code": "code",
            "save_memory": "save_memory",
            "research": "research",
            "tool_node": "tool_node",   # 新增
            "writer": "writer",          # 新增
        },
    )

    # Edges from scheduler (conditional)
    workflow.add_conditional_edges(
        "scheduler",
        route_after_scheduler,
        {
            "validate": "validate",
            "save_memory": "save_memory",
            END: END,
        },
    )

    # Edges from research (conditional)
    workflow.add_conditional_edges(
        "research",
        route_after_research,
        {
            "code": "code",
            END: END,
        },
    )

    # Edges from code (conditional)
    workflow.add_conditional_edges(
        "code",
        route_after_code,
        {
            "code": "code",
            "validate": "validate",
        },
    )

    # Edges from validate (conditional)
    workflow.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "writer": "writer",
            "planning": "planning",       # 新增
            "save_memory": "save_memory",
            "research": "research",
            "advance_subtask": "advance_subtask",
            "code": "code",   # 新增：验证失败重试时回到 code 节点
        },
    )

    # save_memory -> END
    workflow.add_edge("save_memory", END)

    # advance_subtask -> code
    workflow.add_edge("advance_subtask", "code")

    # tool_node -> planning (执行完工具后回到 planning 继续处理)
    workflow.add_edge("tool_node", "planning")
    
    # writer 之后进入 validate（可选）或直接保存
    #workflow.add_edge("writer", "save_memory")
    workflow.add_edge("writer", "validate")   # 或 "save_memory"

    return workflow


def compile_workflow() -> any:
    """Create and compile the workflow graph."""
    workflow = create_workflow()
    return workflow.compile()