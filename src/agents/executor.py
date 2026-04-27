# src/agents/executor.py
import uuid
import time
import tempfile
import os
import re
from typing import Any, Dict, List, Optional

from src.execution.sandbox import CodeSandbox
from src.execution.file_ops import FileOperations
from src.execution.llm_router_pool import get_llm_router_pool
from src.config import config
from src.common.logging import setup_logging
from src.orchestrator.state import AgentState
from src.agents.base import BaseAgent
from src.model_router import get_router

logger = setup_logging("agents.executor")

CODER_SYSTEM_PROMPT = """You are an expert Python programmer. Given a task description,
generate clean, well-documented, and tested Python code.

Rules:
1. Write only valid Python code
2. Include docstrings and comments
3. Handle edge cases and errors
4. Use type hints where appropriate
5. Return the code in a markdown code block with language tag

Output format:
```python
# your code here
```"""


class ExecutorAgent(BaseAgent):
    def __init__(
        self,
        sandbox: Optional[CodeSandbox] = None,
        file_ops: Optional[FileOperations] = None,
        llm_api_url: str = config.llm_api_url,
        llm_model: Optional[str] = None,
    ):
        self.sandbox = sandbox or CodeSandbox()
        self.file_ops = file_ops or FileOperations()
        self.llm_api_url = llm_api_url
        self.llm_model = llm_model or config.llm_model_name

    async def _generate_and_execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("ExecutorAgent starting code generation")
        user_input = context.get("user_input", "")
        research_results = context.get("research_results", [])
        subtasks = context.get("subtasks", [])

        code = await self.generate_with_fallback(user_input, research_results, subtasks)

        if not code:
            logger.error("ExecutorAgent: No code was generated")
            return {
                "code": "",
                "file_path": "",
                "execution_result": {"success": False, "error": "No code generated"},
            }

        file_path = self._write_code_file(code)
        execution_result = await self.sandbox.execute(file_path)

        logger.info(f"ExecutorAgent completed. Success: {execution_result.get('success')}, File: {file_path}")
        return {
            "code": code,
            "file_path": file_path,
            "execution_result": execution_result,
        }

    async def run(self, state: AgentState) -> Dict[str, Any]:
        agent_name = "ExecutorAgent"
        state.step_count += 1
        step = state.step_count
        logger.info(f"Starting {agent_name}, step={step}")
        start_time = time.time()

        context = {
            "user_input": state.user_input,
            "research_results": state.research_results,
            "subtasks": state.subtasks,
        }

        result = await self._generate_and_execute(context)
        duration = time.time() - start_time
        status = "success" if result.get("execution_result", {}).get("success") else "error"
        logger.info(f"{agent_name} completed, step={step}, status={status}, duration={duration:.2f}")
        return {
            "code_generated": result.get("code", ""),
            "code_file_path": result.get("file_path", ""),
            "execution_result": result.get("execution_result"),
            "final_answer": result.get("code", ""),
        }

    async def generate_with_fallback(
        self,
        user_input: str,
        research_results: List[Dict[str, Any]],
        subtasks: List[str],
    ) -> str:
        router = get_router()
        candidates = router.get_candidates(user_input)
        pool = get_llm_router_pool()
        try:
            return await pool.call_with_fallback(
                candidates,
                self._call_llm_for_code,
                user_input, research_results, subtasks
            )
        except Exception as e:
            logger.error(f"All candidate models failed: {e}")
            return ""

    async def _call_llm_for_code(
        self,
        model: str,
        user_input: str,
        research_results: List[Dict[str, Any]],
        subtasks: List[str],
        base_url: Optional[str] = None,
    ) -> str:
        from openai import AsyncOpenAI
        api_url = base_url or self.llm_api_url
        client = AsyncOpenAI(api_key="not-needed", base_url=api_url)

        research_context = self._build_research_context(research_results)
        task_description = f"Task: {user_input}\n\nSubtasks: {'; '.join(subtasks)}"

        if research_context:
            prompt = f"""{task_description}

Research Context:
{research_context}

Please generate Python code that fulfills the above task, using the research context where applicable."""
        else:
            prompt = f"{task_description}\n\nPlease generate Python code that fulfills the above task."

        messages = [
            {"role": "system", "content": CODER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=config.llm_max_tokens,
        )

        raw_output = response.choices[0].message.content or ""
        code = self._extract_code(raw_output)
        if not code:
            raise ValueError("No code extracted from LLM response")
        return code

    def _build_research_context(self, research_results: List[Dict[str, Any]]) -> str:
        if not research_results:
            return ""
        parts = []
        for r in research_results:
            summary = r.get("summary", "")
            source = r.get("source", "unknown")
            parts.append(f"[Source: {source}]\n{summary}")
        return "\n\n---\n\n".join(parts)

    def _extract_code(self, text: str) -> str:
        match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _write_code_file(self, code: str) -> str:
        task_id = uuid.uuid4().hex[:8]
        temp_dir = getattr(self.sandbox, "work_dir", "/tmp/ai_factory")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, f"generated_{task_id}.py")
        with open(file_path, "w") as f:
            f.write(code)
        logger.info(f"Code written to {file_path}")
        return file_path