import uuid
import time
import tempfile
import os
import re
from typing import Any, Dict, List, Optional

from src.execution.sandbox import CodeSandbox
from src.execution.file_ops import FileOperations
from src.execution.llm_pool import LLMPool
from src.execution.tools_registry import ToolsRegistry
from src.config import config
from src.common.logging import setup_logging
from src.common.retry import retry_with_backoff, smart_retry

logger = setup_logging("agents.executor")

PRIMARY_MODEL = "Qwen3.6-35B-A3B-UD-Q5_K_M"
FALLBACK_MODEL = "DeepSeek-R1-Distill-Qwen-32B-Q5_K_M"

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


class ExecutorAgent:
    """Agent responsible for code generation and sandbox execution.

    Supports dual-model fallback, code validation, and tool generation.
    """

    def __init__(
        self,
        sandbox: Optional[CodeSandbox] = None,
        file_ops: Optional[FileOperations] = None,
        llm_api_url: str = config.llm_api_url,
        llm_model: Optional[str] = None,
        llm_pool: Optional[LLMPool] = None,
        tools_registry: Optional[ToolsRegistry] = None,
    ):
        self.sandbox = sandbox or CodeSandbox()
        self.file_ops = file_ops or FileOperations()
        self.llm_api_url = llm_api_url
        self.llm_model = llm_model or config.llm_model_name
        self.llm_pool = llm_pool or LLMPool(
            max_concurrent=config.llm_max_concurrent,
            timeout=config.llm_timeout,
        )
        self.tools_registry = tools_registry or ToolsRegistry(
            tools_dir=config.tools_dir
        )

    async def run(self, context: Dict[str, Any], state: Any = None) -> Dict[str, Any]:
        """Generate code and execute it in sandbox.

        Args:
            context: Dictionary with user_input, research_results, subtasks
            state: Optional LangGraph state object

        Returns:
            Dictionary with code, file_path, execution_result
        """
        logger.info("ExecutorAgent starting code generation")

        user_input = context.get("user_input", "")
        research_results = context.get("research_results", [])
        subtasks = context.get("subtasks", [])

        code = await self.generate_with_fallback(
            user_input, research_results, subtasks
        )

        if not code:
            logger.error("ExecutorAgent: No code was generated")
            return {
                "code": "",
                "file_path": "",
                "execution_result": {"success": False, "error": "No code generated"},
            }

        file_path = self._write_code_file(code)
        execution_result = await self.sandbox.execute(file_path)

        logger.info(
            f"ExecutorAgent completed. Success: {execution_result.get('success')}, "
            f"File: {file_path}"
        )

        return {
            "code": code,
            "file_path": file_path,
            "execution_result": execution_result,
        }

    async def generate_with_fallback(
        self,
        user_input: str,
        research_results: List[Dict[str, Any]],
        subtasks: List[str],
    ) -> str:
        """Generate code using primary model, falling back to secondary model on failure.

        Args:
            user_input: User's task description
            research_results: Research results from KnowledgeRetriever
            subtasks: List of subtasks to complete

        Returns:
            Generated Python code string, or empty string on complete failure
        """
        try:
            logger.info(f"Attempting code generation with primary model: {PRIMARY_MODEL}")
            code = await self._generate_code_with_model(
                model=PRIMARY_MODEL,
                user_input=user_input,
                research_results=research_results,
                subtasks=subtasks,
            )
            if code:
                logger.info("Code generation succeeded with primary model")
                return code
        except Exception as e:
            logger.warning(
                f"Primary model {PRIMARY_MODEL} failed: {e}, "
                f"trying fallback model {FALLBACK_MODEL}"
            )

        try:
            logger.info(f"Attempting code generation with fallback model: {FALLBACK_MODEL}")
            code = await self._generate_code_with_model(
                model=FALLBACK_MODEL,
                user_input=user_input,
                research_results=research_results,
                subtasks=subtasks,
            )
            if code:
                logger.info("Code generation succeeded with fallback model")
                return code
        except Exception as e:
            logger.error(
                f"Fallback model {FALLBACK_MODEL} also failed: {e}"
            )

        return ""

    @smart_retry(max_retries=3, backoff_factor=1.0, fallback_model=FALLBACK_MODEL)
    async def _generate_code_with_model(
        self,
        model: str,
        user_input: str,
        research_results: List[Dict[str, Any]],
        subtasks: List[str],
    ) -> str:
        """Generate code using a specific LLM model.

        Args:
            model: Name of the LLM model to use
            user_input: User's task description
            research_results: Research results
            subtasks: List of subtasks

        Returns:
            Generated Python code string

        Raises:
            Exception: If LLM call fails
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key="not-needed",
            base_url=self.llm_api_url,
        )

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

    def _build_research_context(
        self, research_results: List[Dict[str, Any]]
    ) -> str:
        """Build a context string from research results.

        Args:
            research_results: List of research result dictionaries

        Returns:
            Formatted context string
        """
        if not research_results:
            return ""

        parts = []
        for r in research_results:
            summary = r.get("summary", "")
            source = r.get(
                "source",
                r.get(
                    "sources",
                    [{}],
                )[0].get("source_path", "unknown")
                if r.get("sources")
                else "unknown",
            )
            parts.append(f"[Source: {source}]\n{summary}")

        return "\n\n---\n\n".join(parts)

    def _extract_code(self, text: str) -> str:
        """Extract Python code from markdown code blocks.

        Args:
            text: Text that may contain markdown code blocks

        Returns:
            Extracted code string
        """
        match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        return text.strip()

    def _write_code_file(self, code: str) -> str:
        """Write generated code to a temporary file.

        Args:
            code: Python code string

        Returns:
            Path to the written file
        """
        task_id = uuid.uuid4().hex[:8]
        temp_dir = (
            self.sandbox.work_dir
            if hasattr(self.sandbox, "work_dir")
            else "/tmp/ai_factory"
        )
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, f"generated_{task_id}.py")

        with open(file_path, "w") as f:
            f.write(code)

        logger.info(f"Code written to {file_path}")
        return file_path

    async def validate_code(
        self,
        code: str,
        user_input: str,
        execution_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Validate generated code against the user request.

        Args:
            code: Generated Python code
            user_input: Original user request
            execution_result: Result from sandbox execution

        Returns:
            Dictionary with validation result, score, and feedback
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key="not-needed",
            base_url=self.llm_api_url,
        )

        exec_status = "passed"
        if execution_result:
            if not execution_result.get("success", False):
                exec_status = "failed"
            elif execution_result.get("stderr"):
                exec_status = "errors"

        prompt = f"""Validate the following code against the user request.

User Request:
{user_input}

Generated Code:
```python
{code}
```

Execution Status: {exec_status}
{"Execution Output: " + execution_result.get("stdout", "") if execution_result else ""}
{"Execution Errors: " + execution_result.get("stderr", "") if execution_result else ""}

Please evaluate:
1. Does the code fulfill the user request?
2. Is the code syntactically correct?
3. Does the code handle edge cases?
4. Are there any security concerns?

Return your response in JSON format:
{{
    "passes": true/false,
    "score": 0-10,
    "feedback": "Your explanation"
}}"""

        try:
            response = await client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": CODER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
            )

            raw_output = response.choices[0].message.content or ""
            logger.info(f"Raw LLM response length: {len(raw_output)}")
            logger.info(f"Raw LLM response preview: {raw_output[:500]}")
            
            code = self._extract_code(raw_output)
            logger.info(f"Extracted code length: {len(code)}")
            
            if not code:
                raise ValueError(f"No code extracted from LLM response. Raw output: {raw_output[:200]}")

            validation_text = self._parse_validation_result(raw_output)
            return validation_text

        except Exception as e:
            logger.warning(f"Code validation LLM call failed: {e}")
            return {
                "passes": exec_status == "passed",
                "score": 10 if exec_status == "passed" else 0,
                "feedback": f"Validation failed due to LLM error: {e}",
            }

    def _parse_validation_result(self, text: str) -> Dict[str, Any]:
        """Parse validation result from LLM response.

        Args:
            text: LLM response containing JSON

        Returns:
            Parsed validation result dictionary
        """
        match = re.search(r"\{[^}]*\}", text, re.DOTALL)
        if match:
            try:
                import json

                result = json.loads(match.group())
                return {
                    "passes": result.get("passes", False),
                    "score": result.get("score", 0),
                    "feedback": result.get("feedback", ""),
                }
            except json.JSONDecodeError:
                pass

        return {"passes": False, "score": 0, "feedback": "Could not parse validation result"}

    async def generate_tool_for_agent(
        self,
        tool_name: str,
        tool_description: str,
        parameters: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a tool module for use by other agents.

        Args:
            tool_name: Name of the tool
            tool_description: Description of what the tool does
            parameters: List of parameter schemas
            context: Optional context from research or previous tasks

        Returns:
            Dictionary with tool code, file_path, and registered info
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key="not-needed",
            base_url=self.llm_api_url,
        )

        prompt = f"""You are an expert Python programmer. Create a reusable tool module.

Tool Name: {tool_name}
Tool Description: {tool_description}
Parameters: {parameters}

Requirements:
1. Create a Python module with a `get_tool_info()` function that returns metadata
2. Implement a `run()` function that accepts **kwargs and returns a dict
3. Include proper error handling and validation
4. Add docstrings and type hints
5. Save to: /tmp/ai_factory/tools/{tool_name}.py

Return only the complete Python file content in a markdown code block.

Example structure:
```python
def get_tool_info():
    return {{
        "name": "{tool_name}",
        "description": "{tool_description}",
        "module_path": "/tmp/ai_factory/tools/{tool_name}.py",
        "function_name": "run",
        "parameters": {parameters}
    }}

async def run(**kwargs):
    \"\"\"Execute the tool.\"\"\"
    return {{"success": True, "result": "..."}}
```"""

        if context:
            research = context.get("research_results", [])
            if research:
                research_context = self._build_research_context(research)
                prompt += f"\n\nResearch Context:\n{research_context}"

        try:
            response = await client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": CODER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=config.llm_max_tokens,
            )

            raw_output = response.choices[0].message.content or ""
            tool_code = self._extract_code(raw_output)

            if not tool_code:
                logger.error("No tool code generated")
                return {"success": False, "error": "No tool code generated"}

            tools_dir = self.tools_registry.tools_dir
            os.makedirs(tools_dir, exist_ok=True)
            file_path = os.path.join(tools_dir, f"{tool_name}.py")

            with open(file_path, "w") as f:
                f.write(tool_code)

            self.tools_registry.register_tool(
                name=tool_name,
                description=tool_description,
                module_path=file_path,
                function_name="run",
                parameters={p.get("name", ""): p for p in parameters}
                if parameters
                else {},
            )

            logger.info(f"Generated and registered tool: {tool_name}")

            return {
                "success": True,
                "code": tool_code,
                "file_path": file_path,
                "tool_name": tool_name,
            }

        except Exception as e:
            logger.error(f"Tool generation failed: {e}")
            return {"success": False, "error": str(e)}
