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

    # Current node tracking
    current_node: str = ""

    # Error tracking
    error: str | None = None

    # Additional metadata
    metadata: dict[str, Any] = {}

    def needs_retry(self) -> bool:
        return self.retry_count < self.max_retries and self.error is not None
