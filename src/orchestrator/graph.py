from langgraph.graph import StateGraph, END

from src.orchestrator.state import AgentState
from src.orchestrator.nodes import (
    analyze_node,
    research_node,
    code_node,
    validate_node,
)


def route_after_analyze(state: AgentState) -> str:
    """Decide which node to go to after analysis."""
    if not state.subtasks:
        return END
    # If there are code-related subtasks, go to code; otherwise research
    has_code = any("code" in s.lower() or "implement" in s.lower() for s in state.subtasks)
    return "code" if has_code else "research"


def route_after_research(state: AgentState) -> str:
    """Decide which node to go to after research."""
    # After research, always go to code generation if there are subtasks
    if state.subtasks:
        return "code"
    return END


def route_after_code(state: AgentState) -> str:
    """Decide which node to go to after code execution."""
    if state.execution_result and not state.execution_result.get("success", False):
        if state.needs_retry:
            return "code"
        else:
            return "validate"
    return "validate"


def route_after_validate(state: AgentState) -> str:
    """Decide whether to finish or retry after validation."""
    if state.validation_result and state.validation_result.get("passed", False):
        return END
    if state.needs_retry:
        return "research"
    return END


def create_workflow() -> StateGraph:
    """Build the LangGraph workflow for AI Factory."""

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("research", research_node)
    workflow.add_node("code", code_node)
    workflow.add_node("validate", validate_node)

    # Set entry point
    workflow.set_entry_point("analyze")

    # Edges from analyze (conditional)
    workflow.add_conditional_edges(
        "analyze",
        route_after_analyze,
        {
            "code": "code",
            "research": "research",
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
            END: END,
            "research": "research",
        },
    )

    return workflow


def compile_workflow() -> any:
    """Create and compile the workflow graph."""
    workflow = create_workflow()
    return workflow.compile()
