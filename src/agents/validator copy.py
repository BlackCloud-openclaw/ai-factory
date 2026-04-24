import re
import py_compile
import tempfile
import os
from typing import Any, Optional

from src.config import config
from src.common.logging import setup_logging
from src.common.retry import retry_with_backoff

logger = setup_logging("agents.validator")

VALIDATOR_SYSTEM_PROMPT = """You are a code quality validator. Your job is to verify whether generated code:
1. Correctly implements the user's requirements
2. Is syntactically and logically correct
3. Handles edge cases appropriately
4. Follows Python best practices

Analyze the code carefully against the user request and execution results.
Return your evaluation in strict JSON format with these exact keys:
{
    "passed": true/false,
    "feedback": "detailed explanation of validation result",
    "suggestions": ["list", "of", "improvement", "suggestions"]
}"""


class ValidatorAgent:
    """Agent responsible for validating code quality and requirement fulfillment.

    Performs two levels of validation:
    1. Syntax checking using py_compile
    2. Semantic validation using LLM to assess requirement fulfillment
    """

    def __init__(
        self,
        llm_api_url: str = config.llm_api_url,
        llm_model: str = config.llm_model_name,
    ):
        self.llm_api_url = llm_api_url
        self.llm_model = llm_model

    async def validate(self, code: str, user_input: str, execution_result: Optional[dict] = None) -> dict:
        """Validate code quality and requirement fulfillment.

        Args:
            code: The generated Python code to validate
            user_input: The original user request
            execution_result: Result from code execution (optional)

        Returns:
            dict with keys: passed (bool), feedback (str), suggestions (list)
        """
        logger.info("ValidatorAgent starting validation")

        # Step 1: Syntax check
        syntax_ok, syntax_feedback = self._check_syntax(code)
        if not syntax_ok:
            logger.warning(f"Syntax validation failed: {syntax_feedback}")
            return {
                "passed": False,
                "feedback": f"Syntax error detected: {syntax_feedback}",
                "suggestions": ["Fix the syntax errors before proceeding"],
            }

        # Step 2: LLM-based semantic validation
        llm_result = await self._validate_with_llm(code, user_input, execution_result)

        passed = llm_result.get("passed", False)
        feedback = llm_result.get("feedback", "Validation completed.")
        suggestions = llm_result.get("suggestions", [])

        if passed:
            logger.info("Validation passed")
        else:
            logger.warning(f"Validation failed: {feedback}")
            if suggestions:
                logger.info(f"Suggestions: {suggestions}")

        return {
            "passed": passed,
            "feedback": feedback,
            "suggestions": suggestions,
        }

    def _check_syntax(self, code: str) -> tuple[bool, str]:
        """Check code syntax using py_compile.

        Returns:
            Tuple of (is_valid, feedback_message)
        """
        if not code or not code.strip():
            return False, "Code is empty"

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_path = f.name

            py_compile.compile(temp_path, doraise=True)
            os.unlink(temp_path)
            return True, ""

        except py_compile.PyCompileError as e:
            error_msg = str(e)
            os.unlink(temp_path)
            return False, error_msg
        except Exception as e:
            return False, f"Syntax check error: {e}"

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    async def _validate_with_llm(self, code: str, user_input: str, execution_result: Optional[dict] = None) -> dict:
        """Validate code semantics using LLM.

        Returns:
            dict with keys: passed, feedback, suggestions
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key="not-needed", base_url=self.llm_api_url)

        exec_status = "unknown"
        exec_output = ""
        if execution_result:
            if execution_result.get("success"):
                exec_status = "passed"
                exec_output = execution_result.get("stdout", "")[:500]
            else:
                exec_status = "failed"
                exec_output = execution_result.get("stderr", "")[:500]

        exec_output_section = ""
        if exec_output:
            exec_output_section = "Execution Output:\n" + exec_output
        else:
            exec_output_section = "No execution output available."

        prompt = f"""User Request:
{user_input}

Generated Code:
```python
{code}
```

Execution Status: {exec_status}
{exec_output_section}

Please evaluate whether the code satisfies the user's request.
Return your response in strict JSON format:
{{
    "passed": true/false,
    "feedback": "Detailed explanation of why the code passed or failed validation",
    "suggestions": ["List of specific improvement suggestions, empty list if passed"]
}}"""

        try:
            response = await client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            raw_output = response.choices[0].message.content or ""
            return self._parse_validation_result(raw_output)
        except Exception as e:
            logger.warning(f"LLM validation call failed: {e}, falling back to execution-based check")
            return self._fallback_validation(execution_result)

    def _parse_validation_result(self, text: str) -> dict:
        """Parse validation result from LLM response text.

        Extracts JSON from the response and validates the structure.
        """
        match = re.search(r"\{[^{}]*" + '"passed"' + r"[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                import json
                result = json.loads(match.group())
                return {
                    "passed": bool(result.get("passed", False)),
                    "feedback": str(result.get("feedback", "Validation result parsed.")),
                    "suggestions": result.get("suggestions", []),
                }
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse JSON from LLM response: {e}")

        return {
            "passed": False,
            "feedback": f"Could not parse validation result from LLM. Raw response: {text[:200]}",
            "suggestions": ["Review the LLM response format"],
        }

    def _fallback_validation(self, execution_result: Optional[dict] = None) -> dict:
        """Fallback validation when LLM call fails.

        Uses execution result as the basis for validation.
        """
        if execution_result and execution_result.get("success"):
            return {
                "passed": True,
                "feedback": "Validation passed based on successful execution (LLM unavailable).",
                "suggestions": ["Consider re-running validation when LLM is available"],
            }
        return {
            "passed": False,
            "feedback": "Validation failed: no successful execution and LLM unavailable.",
            "suggestions": ["Fix execution errors and retry"],
        }
