# src/orchestrator/graph.py
import uuid
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

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
from src.orchestrator.phase_resolver import WorkflowPhaseResolver
from src.orchestrator.state_patch import WorkflowPhase
from src.agents.director import DirectorAgent

logger = setup_logging("orchestrator.graph")

# 全局 checkpointer（由 main.py 初始化）
_checkpointer: Optional[BaseCheckpointSaver] = None

def set_checkpointer(checkpointer: BaseCheckpointSaver) -> None:
    """设置全局 checkpointer（在应用启动时调用）"""
    global _checkpointer
    _checkpointer = checkpointer

async def director_node(state: AgentState) -> dict:
    """导演节点：生成叙事蓝图和知识变化序列"""
    agent = DirectorAgent()
    return await agent.run(state)


def route_after_analyze(state: AgentState) -> str:
    # 续写模式：需要检查是否有有效的场景计划
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
    logger.info(f"after_plan: task_type={state.task_type}, outline exists? {state.outline is not None}")
    
    if state.pending_tool_calls:
        return "tool_node"
    
    task_type = getattr(state, 'task_type', 'code')
    
    if task_type == "novel_outline":
        return "save_memory"
    
    if task_type == "scene_plan" and state.scene_plan:
        return "director"
    
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
    phase = WorkflowPhaseResolver.resolve(state)
    logger.info(f"route_after_validate: phase={phase}, current_chapter={state.current_chapter}, "
                f"current_scene_index={state.current_scene_index}, total_scenes={state.total_scenes_in_chapter}")
    if phase == WorkflowPhase.WRITING:
        return "writer"
    if phase == WorkflowPhase.TRANSITIONING:
        return "planning"
    return "save_memory"

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
    workflow.add_node("director", director_node)

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
            "director": "director",
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
    workflow.add_edge("director", "writer")
    workflow.add_edge("writer", "validate")

    return workflow


def compile_workflow() -> any:
    """编译工作流，如果全局 checkpointer 已设置则使用它"""
    workflow = create_workflow()
    if _checkpointer is not None:
        return workflow.compile(checkpointer=_checkpointer)
    else:
        # 回退到无 checkpointer（内存模式）
        logger.warning("No checkpointer set, using in-memory checkpointer")
        return workflow.compile()