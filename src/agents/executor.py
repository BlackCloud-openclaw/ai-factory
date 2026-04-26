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
from src.orchestrator.state import AgentState
from src.agents.base import BaseAgent  # 可选，但推荐
from src.model_router import get_router
from src.execution.llm_router_pool import get_llm_router_pool
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

    async def _generate_and_execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
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
        
    async def run(self, state: AgentState) -> Dict[str, Any]:
        """Unified interface: accept state, return incremental updates."""
        agent_name = "ExecutorAgent"
        state.step_count += 1
        step = state.step_count
        logger.info(f"Starting {agent_name}, step={step}")
        start_time = time.time()
        logger.info("ExecutorAgent starting code generation")

        context = {
            "user_input": state.user_input,
            "research_results": state.research_results,
            "subtasks": state.subtasks,
       }

        # 调用原有的内部方法（需要将 _generate_and_execute 改成直接使用这个 context）
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

    async def _call_llm_for_code(self, model: str, user_input: str, research_results, subtasks) -> str:
        """供池子调用的实际 LLM 请求函数（包装原 _generate_code_with_model 的逻辑）"""
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key="not-needed", base_url=self.llm_api_url)
        research_context = self._build_research_context(research_results)
        task_description = f"Task: {user_input}\n\nSubtasks: {'; '.join(subtasks)}"
        if research_context:
            prompt = f"{task_description}\n\nResearch Context:\n{research_context}\n\nPlease generate Python code."
        else:
            prompt = f"{task_description}\n\nPlease generate Python code."
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
        
    async def generate_with_fallback(self, user_input: str, research_results, subtasks) -> str:
        router = get_router()
        candidates = router.get_candidates(user_input)
        pool = get_llm_router_pool()
        last_exception = None
        for model in candidates:
            try:
                logger.info(f"Attempting code generation with model: {model}")
                code = await pool.call(model, self._call_llm_for_code, user_input, research_results, subtasks)
                if code:
                    logger.info(f"Code generation succeeded with model: {model}")
                    return code
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}, trying next candidate")
                last_exception = e
        logger.error(f"All candidate models failed: {last_exception}")
        return ""

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
