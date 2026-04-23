from langgraph.graph import StateGraph, END

from src.orchestrator.state import AgentState
from src.orchestrator.nodes import (
    analyze_node,
    research_node,
    code_node,
    validate_node,
    load_memory_node,
    save_memory_node,
    plan_node,
    advance_subtask_node,
)


def route_after_analyze(state: AgentState) -> str:
    """Decide which node to go to after analysis."""
    subtasks = state.subtasks if hasattr(state, 'subtasks') else state.get('subtasks', [])
    if not subtasks:
        return END
    has_code = any('code' in s.lower() or 'implement' in s.lower() for s in subtasks)
    return 'code' if has_code else 'research'


def route_after_plan(state: AgentState) -> str:
    """Decide which node to go to after planning based on first subtask type."""
    plan = getattr(state, 'plan', None)
    if not plan:
        return END
    first_subtask = plan[0]
    subtask_type = first_subtask.get("type", "research")
    route_map = {
        "research": "research",
        "code": "code",
        "validate": "validate",
        "write": "code",
        "plan": "research",
    }
    return route_map.get(subtask_type, "research")


def route_after_research(state: AgentState) -> str:
    subtasks = state.subtasks if hasattr(state, 'subtasks') else state.get('subtasks', [])
    return 'code' if subtasks else END


def route_after_code(state: AgentState) -> str:
    exec_result = state.execution_result if hasattr(state, 'execution_result') else state.get('execution_result', {})
    if exec_result and not exec_result.get('success', False):
        needs_retry = state.needs_retry if hasattr(state, 'needs_retry') else state.get('needs_retry', False)
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


def create_workflow() -> StateGraph:
    """Build the LangGraph workflow for AI Factory."""

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("load_memory", load_memory_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("planning", plan_node)
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
        route_after_plan,
        {
            "research": "research",
            "code": "code",
            "validate": "validate",
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
