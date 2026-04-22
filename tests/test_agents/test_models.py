import pytest
from src.common.models import AgentState, Message, ExecutionResult, KnowledgeSearchResult, AgentResponse


class TestAgentState:
    def test_default_state(self):
        state = AgentState()
        assert state.user_input == ""
        assert state.messages == []
        assert state.intent == ""
        assert state.subtasks == []
        assert state.retry_count == 0
        assert state.max_retries == 3

    def test_custom_state(self):
        state = AgentState(
            user_input="Write a function",
            intent="code_generation",
            subtasks=["write function", "add tests"],
            max_retries=5,
        )
        assert state.user_input == "Write a function"
        assert state.intent == "code_generation"
        assert state.subtasks == ["write function", "add tests"]
        assert state.max_retries == 5

    def test_needs_retry_true(self):
        state = AgentState(error="something failed", retry_count=0, max_retries=3)
        assert state.needs_retry() is True

    def test_needs_retry_false_no_error(self):
        state = AgentState(error=None, retry_count=0)
        assert state.needs_retry() is False

    def test_needs_retry_false_max_retries(self):
        state = AgentState(error="failed", retry_count=3, max_retries=3)
        assert state.needs_retry() is False


class TestMessage:
    def test_message_creation(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"


class TestExecutionResult:
    def test_successful_execution(self):
        result = ExecutionResult(success=True, stdout="output", returncode=0)
        assert result.success is True
        assert result.stdout == "output"
        assert result.returncode == 0

    def test_failed_execution(self):
        result = ExecutionResult(
            success=False,
            stderr="error occurred",
            returncode=1,
        )
        assert result.success is False
        assert result.stderr == "error occurred"


class TestKnowledgeSearchResult:
    def test_result_creation(self):
        result = KnowledgeSearchResult(
            chunk_id="abc123",
            content="test content",
            document_id="doc-1",
            score=0.95,
        )
        assert result.chunk_id == "abc123"
        assert result.score == 0.95


class TestAgentResponse:
    def test_successful_response(self):
        response = AgentResponse(success=True, answer="Done!")
        assert response.success is True
        assert response.answer == "Done!"

    def test_failed_response(self):
        response = AgentResponse(success=False, error="Something went wrong")
        assert response.success is False
        assert response.error == "Something went wrong"
