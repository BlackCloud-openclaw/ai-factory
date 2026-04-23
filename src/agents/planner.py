import json
import logging
from typing import List, Dict, Any
from src.common.models import llm_call

logger = logging.getLogger(__name__)

PLANNER_PROMPT = """You are a planner agent. Break down the user's request into a sequence of subtasks.

User request: {user_input}
Memory context: {memory_context}

Return a JSON array of subtasks, each with:
- id: unique string (e.g., "task1")
- type: one of ["research", "code", "validate", "write", "plan"]
- description: clear description of what to do
- depends_on: list of subtask ids that must be completed before this one (can be empty)

Example:
[
  {{"id": "task1", "type": "research", "description": "Search for async best practices", "depends_on": []}},
  {{"id": "task2", "type": "code", "description": "Write async fetch function", "depends_on": ["task1"]}},
  {{"id": "task3", "type": "validate", "description": "Check code for errors", "depends_on": ["task2"]}}
]

Only output valid JSON, no other text.
"""


def plan_task(user_input: str, memory_context: str = "") -> List[Dict[str, Any]]:
    """Call LLM to generate subtask list."""
    prompt = PLANNER_PROMPT.format(user_input=user_input, memory_context=memory_context)
    response = llm_call(prompt, temperature=0.3)
    try:
        data = json.loads(response)
        if not isinstance(data, list):
            raise ValueError("Response is not a list")
        for item in data:
            if not all(k in item for k in ("id", "type", "description", "depends_on")):
                raise ValueError(f"Missing field in subtask: {item}")
        return data
    except Exception as e:
        logger.error(f"Failed to parse planner output: {e}\nResponse: {response}")
        return [
            {
                "id": "fallback",
                "type": "code",
                "description": user_input,
                "depends_on": [],
            }
        ]
