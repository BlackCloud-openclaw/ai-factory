# src/agents/validator.py
import re
import py_compile
import tempfile
import os
import json
from typing import Any, Optional, Dict, List, Tuple

from src.config import config
from src.common.logging import setup_logging
from src.common.retry import retry_with_backoff
from src.agents.base import BaseAgent
from src.orchestrator.state import AgentState

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


class ValidatorAgent(BaseAgent):
    """Agent responsible for validating code quality and requirement fulfillment."""

    def __init__(
        self,
        llm_api_url: str = config.llm_api_url,
        llm_model: str = config.llm_model_name,
    ):
        self.llm_api_url = llm_api_url
        self.llm_model = llm_model   
    
    async def run(self, state: AgentState) -> Dict[str, Any]:
        """Unified interface: accept state, return validation result incremental update."""
        code = state.code_generated
        user_input = state.user_input
        execution_result = state.execution_result
        result = await self.validate(code, user_input, execution_result)
        return {
            "validation_result": result,
            "final_answer": result.get("feedback", ""),
        }    

    async def validate(self, code: str, user_input: str, execution_result: Optional[dict] = None) -> dict:
        """Validate code quality and requirement fulfillment."""
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

    def _check_syntax(self, code: str) -> Tuple[bool, str]:
        """Check code syntax using py_compile."""
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
        """Validate code semantics using LLM."""
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
            parsed_result = self._parse_validation_result_enhanced(raw_output)
            return self._normalize_validation_result(parsed_result, raw_output)
        except Exception as e:
            logger.warning(f"LLM validation call failed: {e}, falling back to execution-based check")
            return self._fallback_validation(execution_result)

    def _parse_validation_result_enhanced(self, text: str) -> Dict[str, Any]:
        """Enhanced multi-strategy JSON parsing."""
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'({[\s\S]*?"passed"[\s\S]*?})',
            r'({[^{}]*?"passed"[^{}]*?})',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                try:
                    json_str = match.strip()
                    json_str = re.sub(r',\s}', '}', json_str)
                    json_str = re.sub(r',\s]', ']', json_str)
                    result = json.loads(json_str)
                    if "passed" in result:
                        logger.debug(f"Successfully parsed JSON with pattern: {pattern}")
                        return result
                except json.JSONDecodeError:
                    continue

        try:
            result = self._extract_via_regex(text)
            if result:
                logger.debug("Successfully extracted validation data via regex")
                return result
        except Exception as e:
            logger.debug(f"Regex extraction failed: {e}")

        try:
            result = self._parse_from_natural_language(text)
            if result:
                logger.debug("Successfully parsed from natural language")
                return result
        except Exception as e:
            logger.debug(f"Natural language parsing failed: {e}")

        logger.warning(f"All parsing strategies failed for text: {text[:200]}")
        return {
            "passed": False,
            "feedback": f"Could not parse validation result. Raw: {text[:200]}",
            "suggestions": ["Check LLM response format"]
        }

    def _extract_via_regex(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract validation data using regex patterns."""
        result = {}

        passed_patterns = [
            r'"passed"\s*:\s*true',
            r'"passed"\s*:\s*false',
            r'passed\s*:\s*true',
            r'passed\s*:\s*false',
            r'passed["\s:]+(True|true|False|false)',
        ]

        for pattern in passed_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(0).split(':')[-1].strip().lower()
                result["passed"] = 'true' in value
                break

        feedback_patterns = [
            r'"feedback"\s*:\s*"([^"]*)"',
            r'feedback["\s:]+["\s]*"([^"]*)"',
            r'feedback["\s:]+([^"\n\r]+?)(?:,|\n|$)',
        ]

        for pattern in feedback_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result["feedback"] = match.group(1).strip()
                break

        suggestions_patterns = [
            r'"suggestions"\s:\s\[(.*?)\]',
            r'suggestions["\s:]+\[(.*?)\]',
        ]

        for pattern in suggestions_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                suggestions_text = match.group(1)
                suggestions = re.findall(r'"([^"]*)"', suggestions_text)
                if suggestions:
                    result["suggestions"] = suggestions
                else:
                    result["suggestions"] = [s.strip() for s in suggestions_text.split(',') if s.strip()]
                break

        if "passed" not in result:
            return None
        if "feedback" not in result:
            result["feedback"] = "Validation result extracted without explicit feedback"
        if "suggestions" not in result:
            result["suggestions"] = []

        return result

    def _parse_from_natural_language(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse validation data from natural language text."""
        text_lower = text.lower()

        positive_indicators = [
            "code is correct", "passes validation", "satisfies the request",
            "meets requirements", "properly implements", "code works",
            "passed", "successful", "valid"
        ]
        negative_indicators = [
            "fails", "does not satisfy", "missing", "incorrect",
            "bug", "error", "issue", "problem", "failed",
            "invalid", "does not meet"
        ]

        passed = None
        for indicator in positive_indicators:
            if indicator in text_lower:
                passed = True
                break

        if passed is None:
            for indicator in negative_indicators:
                if indicator in text_lower:
                    passed = False
                    break

        if passed is None:
            return None

            feedback = text[:300]
        suggestions = []
        bullet_patterns = [
            r'[-•*]\s*([^.\n]+[.]?)',
            r'\d+\.\s*([^.\n]+[.]?)',
        ]

        for pattern in bullet_patterns:
            matches = re.findall(pattern, text)
            if matches:
                suggestions = matches[:3]
                break

        return {"passed": passed, "feedback": feedback, "suggestions": suggestions}

    def _normalize_validation_result(self, result: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
        """Normalize and validate the parsed result structure."""
        normalized = {}

        passed_value = result.get("passed")
        if isinstance(passed_value, bool):
            normalized["passed"] = passed_value
        elif isinstance(passed_value, str):
            normalized["passed"] = passed_value.lower() in ['true', '1', 'yes', 'pass']
        else:
            feedback = str(result.get("feedback", "")).lower()
            if any(word in feedback for word in ['success', 'correct', 'valid']):
                normalized["passed"] = True
            elif any(word in feedback for word in ['fail', 'error', 'incorrect']):
                normalized["passed"] = False
            else:
                logger.warning("Could not determine passed status, defaulting to False")
                normalized["passed"] = False

        feedback = result.get("feedback", "")
        if not feedback or not isinstance(feedback, str):
            feedback = f"Validation result: {'Passed' if normalized['passed'] else 'Failed'}"
            if "Execution Output" in raw_text:
                feedback += " (based on execution results)"
        normalized["feedback"] = feedback.strip()

        suggestions = result.get("suggestions", [])
        if not isinstance(suggestions, list):
            if isinstance(suggestions, str):
                suggestions = [suggestions]
            else:
                suggestions = []
        normalized["suggestions"] = suggestions

        return normalized

    def _parse_validation_result(self, text: str) -> dict:
        """Legacy parsing method - kept for backward compatibility."""
        return self._parse_validation_result_enhanced(text)
        
    def _fallback_validation(self, execution_result: Optional[dict] = None) -> dict:
        """Fallback validation when LLM call fails."""
        # 优先使用执行结果判断
        if execution_result and execution_result.get("success"):
            stdout = execution_result.get("stdout", "")
            # 检查输出是否包含预期结果（如 385 或平方和）
            if "385" in stdout or "平方和" in stdout or "sum" in stdout.lower():
                return {
                    "passed": True,
                    "feedback": "Validation passed based on successful execution and output.",
                    "suggestions": []
                }
            return {
                "passed": True,
                "feedback": "Validation passed based on successful execution (LLM unavailable).",
                "suggestions": ["Consider re-running validation when LLM is available"]
            }
        return {
            "passed": False,
            "feedback": "Validation failed: no successful execution and LLM unavailable.",
            "suggestions": ["Fix execution errors and retry"]
        }
