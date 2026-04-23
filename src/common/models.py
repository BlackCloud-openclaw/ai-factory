import os
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from openai import OpenAI
from src.config import config
import logging
import httpx

logger = logging.getLogger(__name__)

# Disable proxy environment variables to prevent LLM connection issues
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']:
    os.environ.pop(key, None)


class NodeName(str, Enum):
    ANALYZE = "analyze"
    RESEARCH = "research"
    CODE = "code"
    VALIDATE = "validate"


class AgentState(BaseModel):
    """LangGraph state for the AI Factory orchestrator."""

    user_input: str = ""
    messages: List[Dict[str, str]] = Field(default_factory=list)
    intent: str = ""
    subtasks: List[str] = Field(default_factory=list)
    research_results: List[Dict[str, Any]] = Field(default_factory=list)
    code_generated: str = ""
    code_file_path: str = ""
    execution_result: Optional[Dict[str, Any]] = None
    validation_result: Optional[Dict[str, Any]] = None
    final_answer: str = ""
    retry_count: int = 0
    max_retries: int = 3
    current_node: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def needs_retry(self) -> bool:
        return self.retry_count < self.max_retries and self.error is not None


class Message(BaseModel):
    role: str
    content: str


class ExecutionResult(BaseModel):
    success: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    execution_time_ms: float = 0.0
    file_path: str = ""


class KnowledgeSearchResult(BaseModel):
    chunk_id: str
    content: str
    document_id: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    success: bool
    answer: str = ""
    research_used: bool = False
    code_executed: bool = False
    execution_result: Optional[Dict[str, Any]] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None


def llm_call(prompt: str, temperature: float = 0.3, max_tokens: int = 4096) -> str:
    """Synchronous LLM call using the configured API.

    Args:
        prompt: The text prompt to send to the LLM.
        temperature: Sampling temperature (default 0.3 for deterministic output).
        max_tokens: Maximum tokens in the response.

    Returns:
        The LLM's response text.
    """
    try:
        client = OpenAI(
            base_url=config.llm_api_url,
            api_key="not-needed",
            http_client=httpx.Client(proxy=None),
        )
        response = client.chat.completions.create(
            model=config.llm_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ""
