import uuid
import time
import tempfile
import os
from typing import Any, Optional

from src.execution.sandbox import CodeSandbox
from src.execution.file_ops import FileOperations
from src.config import config
from src.common.logging import setup_logging
from src.common.retry import retry_with_backoff

logger = setup_logging("agents.coder")

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


class CoderAgent:
    """Agent responsible for code generation and sandbox execution."""

    def __init__(
        self,
        sandbox: Optional[CodeSandbox] = None,
        file_ops: Optional[FileOperations] = None,
        llm_api_url: str = config.llm_api_url,
        llm_model: str = config.llm_model_name,
    ):
        self.sandbox = sandbox or CodeSandbox()
        self.file_ops = file_ops or FileOperations()
        self.llm_api_url = llm_api_url
        self.llm_model = llm_model

    async def run(self, context: dict[str, Any], state: Any = None) -> dict[str, Any]:
        """Generate code and execute it in sandbox."""
        logger.info("CoderAgent starting code generation")

        user_input = context.get("user_input", "")
        research_results = context.get("research_results", [])
        subtasks = context.get("subtasks", [])

        # Step 1: Generate code via LLM
        code = await self._generate_code(user_input, research_results, subtasks)

        if not code:
            logger.error("CoderAgent: No code was generated")
            return {
                "code": "",
                "file_path": "",
                "execution_result": {"success": False, "error": "No code generated"},
            }

        # Step 2: Write code to file
        file_path = self._write_code_file(code)

        # Step 3: Execute in sandbox
        execution_result = await self.sandbox.execute(file_path)

        logger.info(
            f"CoderAgent completed. Success: {execution_result.get('success')}, "
            f"File: {file_path}"
        )

        return {
            "code": code,
            "file_path": file_path,
            "execution_result": execution_result,
        }

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def _generate_code(
        self,
        user_input: str,
        research_results: list[dict[str, Any]],
        subtasks: list[str],
    ) -> str:
        """Call LLM to generate Python code."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key="not-needed",
            base_url=self.llm_api_url,
        )

        # Build context from research results
        research_context = ""
        if research_results:
            parts = []
            for r in research_results:
                summary = r.get("summary", "")
                source = r.get("source", r.get("sources", [{}])[0].get("source_path", "unknown") if r.get("sources") else "unknown")
                parts.append(f"[Source: {source}]\n{summary}")
            research_context = "\n\n---\n\n".join(parts)

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
            model=self.llm_model,
            messages=messages,
            temperature=0.2,
            max_tokens=config.llm_max_tokens,
        )

        raw_output = response.choices[0].message.content or ""
        return self._extract_code(raw_output)

    def _extract_code(self, text: str) -> str:
        """Extract Python code from markdown code blocks."""
        import re

        # Try to extract from markdown code block
        match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # If no code block found, return the whole text
        return text.strip()

    def _write_code_file(self, code: str) -> str:
        """Write generated code to a temporary file."""
        task_id = uuid.uuid4().hex[:8]
        file_path = os.path.join(
            config.postgres_db,  # Will be overridden by FileOperations
            f"generated_{task_id}.py",
        )

        # Use temp directory for sandboxed execution
        temp_dir = self.sandbox.work_dir if hasattr(self.sandbox, "work_dir") else "/tmp/ai_factory"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, f"generated_{task_id}.py")

        with open(file_path, "w") as f:
            f.write(code)

        logger.info(f"Code written to {file_path}")
        return file_path
