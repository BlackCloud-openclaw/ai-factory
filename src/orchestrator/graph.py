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
    return "planning"


def route_after_plan(state: AgentState) -> str:
    """Decide which node to go to after planning.

    Routes to scheduler for complex multi-subtask plans, or to individual
    nodes for simple single-subtask plans.
    """
    # Check new task_plan field first (from graph.py plan_node)
    plan = getattr(state, 'task_plan', None)
    if not plan:
        # Fallback to legacy plan field (from nodes.py plan_node)
        plan = getattr(state, 'plan', None)
    if not plan:
        return END

    # If plan is a dict (new format with subtasks list)
    if isinstance(plan, dict):
        subtasks = plan.get("subtasks", [])
        if len(subtasks) > 1:
            return "scheduler"
        elif len(subtasks) == 1:
            subtask_type = subtasks[0].get("type", "code")
            route_map = {
                "research": "research",
                "code": "code",
                "validate": "validate",
                "write": "code",
                "plan": "code",
            }
            return route_map.get(subtask_type, "code")
        else:
            return END
    # Legacy format (list of subtask dicts)
    elif isinstance(plan, list):
        if len(plan) > 1:
            return "scheduler"
        elif len(plan) == 1:
            subtask_type = plan[0].get("type", "code")
            route_map = {
                "research": "research",
                "code": "code",
                "validate": "validate",
                "write": "code",
                "plan": "code",
            }
            return route_map.get(subtask_type, "code")
        else:
            return END

    return END


def after_plan(state: AgentState) -> str:
    # plan_node already executed all subtasks via TaskScheduler
    return "save_memory"


def route_after_scheduler(state: AgentState) -> str:
    """Decide which node to go to after scheduler completes.

    Routes to validate if there's code output, otherwise to save_memory.
    """
    plan_status = getattr(state, 'plan_status', '')
    code_generated = getattr(state, 'code_generated', '') or ''

    if plan_status == "failed":
        return 'save_memory'
    elif code_generated and len(code_generated.strip()) > 10:
        return 'validate'
    else:
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
    # 获取验证结果
    if hasattr(state, 'validation_result'):
        validation_result = state.validation_result
    else:
        validation_result = state.get('validation_result', {}) if hasattr(state, 'get') else {}
    validation_passed = validation_result.get('passed', False) if validation_result else False
    
    # 获取重试次数
    retry_count = getattr(state, 'retry_count', 0)
    max_retries = getattr(state, 'max_retries_per_subtask', 2)
    
    # 获取剩余子任务
    remaining = getattr(state, 'remaining_subtasks', [])
    
    if validation_passed:
        return 'advance_subtask' if remaining else 'save_memory'
    else:
        return 'research' if retry_count < max_retries else 'save_memory'


async def plan_node(state: AgentState) -> Dict[str, Any]:
    """Plan complex tasks using PlannerAgent and TaskScheduler.

    Calls PlannerAgent to generate a TaskPlan, submits the plan to
    TaskScheduler, runs all subtasks, and returns the results.
    """
    logger.info(f"Planning task: {state.original_request[:200] if hasattr(state, 'original_request') else state.user_input[:200]}...")

    planner = PlannerAgent()
    task_plan = await planner.plan(
        state.original_request if hasattr(state, "original_request") else state.user_input,
        {},
    )
    scheduler = TaskScheduler()
    await scheduler.submit_plan(task_plan)
    results = await scheduler.run()

    return {
    "task_plan": task_plan.dict() if hasattr(task_plan, "dict") else task_plan.model_dump(),
    "subtask_results": results,
}


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
            END: END,
        },
    )

    # Edges from planning (conditional)
    workflow.add_conditional_edges(
        "planning",
        after_plan,
        {
            "save_memory": "save_memory",
            "research": "research",
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
            "save_memory": "save_memory",
            "research": "research",
            "advance_subtask": "advance_subtask",
        },
    )

    # save_memory -> END
    workflow.add_edge("save_memory", END)

    # advance_subtask -> code
    workflow.add_edge("advance_subtask", "code")

    return workflow


def compile_workflow() -> any:
    """Create and compile the workflow graph."""
    workflow = create_workflow()
    return workflow.compile()
