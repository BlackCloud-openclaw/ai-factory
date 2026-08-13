# src/orchestrator/graph.py
"""
LangGraph 工作流定义
"""

from functools import partial
from typing import Optional, Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from src.orchestrator.state import AgentState
from src.orchestrator.nodes import (
    load_memory_node,
    analyze_node,
    plan_node,
    scheduler_node,
    research_node,
    code_node,
    validate_node,
    save_memory_node,
    advance_subtask_node,
    tool_node_v2,
    writer_node,
    rewrite_node,
    drama_planner_node,
)
from src.orchestrator.phase_resolver import WorkflowPhaseResolver
from src.writing.bootstrap.composition_root import WriterRuntime, build_writer_runtime
from src.config import config
from src.common.logging import setup_logging
from src.orchestrator.state_patch import WorkflowPhase

logger = setup_logging("orchestrator.graph")

# 全局 checkpointer
_checkpointer: Optional[BaseCheckpointSaver] = None


def set_checkpointer(checkpointer: BaseCheckpointSaver) -> None:
    """设置全局 checkpointer（在应用启动时调用）"""
    global _checkpointer
    _checkpointer = checkpointer


# ---------- 路由函数 ----------

def route_after_analyze(state: AgentState) -> str:
    """分析节点后的路由"""
    if state.resume:
        scene_plan_list = getattr(state, 'scene_plan_list', [])
        current_scene_index = getattr(state, 'current_scene_index', 0)
        if scene_plan_list and current_scene_index < len(scene_plan_list):
            return "writer"
        else:
            return "planning"

    task_type = getattr(state, 'task_type', 'code')
    if task_type == "scene_plan" and state.scene_plan:
        return "writer"
    if task_type == "novel_outline" and state.outline:
        return "save_memory"
    return "planning"


def after_plan(state: AgentState) -> str:
    """规划节点后的路由"""
    logger.info(f"after_plan: task_type={state.task_type}, outline exists? {state.outline is not None}")

    # Phase 13.2.2 审计：检查 planner_outputs 状态
    planner_count = len(state.planner_outputs or [])
    metadata_count = len(state.metadata.get("planner_outputs", []))
    logger.info(
        f"after_plan state audit: planner_outputs count={planner_count}, "
        f"metadata count={metadata_count}"
    )

    if state.pending_tool_calls:
        return "tool_node"

    task_type = getattr(state, 'task_type', 'code')

    if task_type == "novel_outline":
        return "save_memory"

    if task_type == "scene_plan" and state.scene_plan:
        if state.scene_plan_list and state.scene_plan_list[0].get("drama"):
            return "writer"
        return "drama_planner"

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
        if state.should_retry():
            return 'code'
    return 'validate'


def route_after_validate(state: AgentState) -> str:
    phase = WorkflowPhaseResolver.resolve(state)
    logger.info(f"route_after_validate: phase={phase}, current_chapter={state.current_chapter}, "
                f"current_scene_index={state.current_scene_index}, total_scenes={state.total_scenes_in_chapter}")
    if phase == WorkflowPhase.WRITING:
        return "writer"
    if phase == WorkflowPhase.TRANSITIONING:
        return "planning"
    if phase == WorkflowPhase.PLANNING:
        return "planning"
    return "save_memory"


def route_after_writer(state: AgentState) -> str:
    if config.experiment_enable_versioned_writer:
        return "validate"
    return "rewrite"


# ---------- 创建工作流 ----------

def create_workflow(runtime: Optional[WriterRuntime] = None) -> StateGraph:
    """
    创建工作流图。

    Args:
        runtime: WriterRuntime，如果未提供则使用默认构建

    Returns:
        StateGraph: 配置好的工作流图
    """
    if runtime is None:
        runtime = build_writer_runtime()

    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("load_memory", load_memory_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("planning", plan_node)
    workflow.add_node("scheduler", scheduler_node)
    workflow.add_node("research", research_node)
    workflow.add_node("code", code_node)

    # 🔑 关键：绑定 runtime 到 validate_node 和 writer_node
    workflow.add_node("validate", partial(validate_node, runtime=runtime))
    workflow.add_node("writer", partial(writer_node, runtime=runtime))

    workflow.add_node("save_memory", save_memory_node)
    workflow.add_node("advance_subtask", advance_subtask_node)
    workflow.add_node("tool_node", tool_node_v2)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("drama_planner", drama_planner_node)

    # 设置入口
    workflow.set_entry_point("load_memory")
    workflow.add_edge("load_memory", "analyze")

    # 条件边
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
            "drama_planner": "drama_planner",
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

    workflow.add_conditional_edges(
        "writer",
        route_after_writer,
        {
            "rewrite": "rewrite",
            "validate": "validate",
        }
    )

    # 固定边
    workflow.add_edge("save_memory", END)
    workflow.add_edge("advance_subtask", "code")
    workflow.add_edge("tool_node", "planning")
    workflow.add_edge("drama_planner", "writer")

    return workflow


def compile_workflow(runtime: Optional[WriterRuntime] = None) -> Any:
    """
    编译工作流。

    Args:
        runtime: WriterRuntime，如果未提供则使用默认构建

    Returns:
        CompiledGraph: 编译后的工作流
    """
    workflow = create_workflow(runtime)
    if _checkpointer is not None:
        return workflow.compile(checkpointer=_checkpointer)
    else:
        logger.warning("No checkpointer set, using in-memory checkpointer")
        return workflow.compile()