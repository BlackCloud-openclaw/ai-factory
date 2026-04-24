from typing import Annotated, Any, Sequence
from langgraph.graph.message import add_messages
from pydantic import BaseModel


class AgentState(BaseModel):
    """State definition for the AI Factory LangGraph workflow."""

    # User input
    user_input: str = ""

    # Message history (accumulated via add_messages)
    messages: Annotated[list, add_messages] = []

    # Analysis results
    intent: str = ""
    subtasks: list[str] = []

    # Research results from knowledge base
    research_results: list[dict[str, Any]] = []

    # Code generation results
    code_generated: str = ""
    code_file_path: str = ""

    # Execution results
    execution_result: dict[str, Any] | None = None

    # Validation results
    validation_result: dict[str, Any] | None = None

    # Final output
    final_answer: str = ""

    # Retry control
    retry_count: int = 0
    max_retries: int = 3
    remaining_subtasks: list = []
    max_retries_per_subtask: int = 3

    # Current node tracking
    current_node: str = ""

    # Error tracking
    error: str | None = None

    # Additional metadata
    metadata: dict[str, Any] = {}

    # Project identifier for memory isolation
    project_id: str = ""

    # Memory context loaded from MemoryAgent
    memory_context: dict[str, Any] = {}

    # Complexity flag and plan execution state
    is_complex: bool = False

    # Skip remaining nodes after plan (set by plan_node)
    skip_remaining: bool = False
    plan: list[dict[str, Any]] = []
    current_subtask_index: int = 0
    current_subtask_id: str = ""
    
    # Scheduler fields
    task_id: str = ""
    subtask_results: dict[str, Any] = {}
    plan_status: str = ""
    task_plan: dict[str, Any] | None = None
    
    # Retry flag (set by nodes, checked by routing functions)
    needs_retry: bool = False
    
    def should_retry(self) -> bool:
        return self.retry_count < self.max_retries and self.error is not None
