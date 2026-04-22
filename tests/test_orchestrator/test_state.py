import pytest
from src.orchestrator.state import AgentState


class TestOrchestratorState:
    def test_state_has_all_fields(self):
        state = AgentState()
        # Check that all expected fields exist
        assert hasattr(state, "user_input")
        assert hasattr(state, "messages")
        assert hasattr(state, "intent")
        assert hasattr(state, "subtasks")
        assert hasattr(state, "research_results")
        assert hasattr(state, "code_generated")
        assert hasattr(state, "execution_result")
        assert hasattr(state, "validation_result")
        assert hasattr(state, "final_answer")
        assert hasattr(state, "retry_count")
        assert hasattr(state, "max_retries")
        assert hasattr(state, "current_node")
        assert hasattr(state, "error")
        assert hasattr(state, "metadata")

    def test_model_copy(self):
        state = AgentState(user_input="original", intent="test")
        new_state = state.model_copy(update={"user_input": "updated"})
        assert new_state.user_input == "updated"
        assert new_state.intent == "test"
        assert state.user_input == "original"  # original unchanged

    def test_needs_retry_property(self):
        state = AgentState(error="fail", retry_count=0, max_retries=3)
        assert state.needs_retry() is True

        state.retry_count = 3
        assert state.needs_retry() is False

        state.error = None
        assert state.needs_retry() is False
