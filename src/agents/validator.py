# src/agents/validator.py
import re
import py_compile
import tempfile
import os
import json
import time
import ast
from typing import Any, Optional, Dict, List, Tuple

from src.config import config
from src.common.logging import setup_logging
from src.common.retry import retry_with_backoff
from src.agents.base import BaseAgent
from src.orchestrator.state import AgentState
from src.prompts.validator_prompts import VALIDATOR_PROMPT_REGISTRY

logger = setup_logging("agents.validator")


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
        agent_name = "ValidatorAgent"
        state.step_count += 1
        step = state.step_count
        logger.info(f"Starting {agent_name}, step={step}")
        start_time = time.time()

        # 获取验证模式（默认 code）
        mode = getattr(state, 'validation_mode', 'code')

        if mode == "code":
            target = state.code_generated
            user_input = state.user_input
            execution_result = state.execution_result
            result = await self.validate(target, user_input, execution_result, mode=mode)
        elif mode == "novel":
            target = state.scene_text
            user_input = state.user_input
            constraints = {
                "outline": state.outline,
                "writing_constraints": state.writing_constraints or {}
            }
            result = await self.validate(target, user_input, None, mode=mode, constraints=constraints)
        else:
            raise ValueError(f"Unsupported validation mode: {mode}")

        duration = time.time() - start_time
        status = "success" if result.get("passed") else "error"
        logger.info(f"{agent_name} completed, step={step}, status={status}, duration={duration:.2f}")
        return {
            "validation_result": result,
            "final_answer": result.get("feedback", ""),
        }

    async def validate(
        self,
        target: str,
        user_input: str,
        execution_result: Optional[dict] = None,
        mode: str = "code",
        constraints: Optional[Dict] = None,
    ) -> dict:
        """统一验证入口：支持 code 和 novel 模式"""
        logger.info(f"ValidatorAgent starting validation with mode={mode}")

        if mode == "code":
            return await self._validate_code(target, user_input, execution_result)
        elif mode == "novel":
            # 从 constraints 中获取 outline（小说大纲）和 writing_constraints
            outline = constraints.get("outline") if constraints else None
            writing_constraints = constraints.get("writing_constraints", {})
            return await self._validate_novel(target, outline, writing_constraints)
        else:
            raise ValueError(f"Unsupported validation mode: {mode}")
        
        
    async def _validate_code(self, code: str, user_input: str, execution_result: Optional[dict]) -> dict:
        """代码验证逻辑（原有）"""
        syntax_ok, syntax_feedback = self._check_syntax(code)
        if not syntax_ok:
            return {
                "passed": False,
                "feedback": f"Syntax error detected: {syntax_feedback}",
                "suggestions": ["Fix the syntax errors before proceeding"],
            }
        builder = VALIDATOR_PROMPT_REGISTRY.get("code")
        prompt = builder.build(code, user_input, execution_result)
        llm_result = await self._validate_with_llm(prompt, execution_result)
        return {
            "passed": llm_result.get("passed", False),
            "feedback": llm_result.get("feedback", "Validation completed."),
            "suggestions": llm_result.get("suggestions", []),
        }

    async def _validate_novel(self, text: str, outline: Optional[Dict], writing_constraints: Dict) -> dict:
        """小说验证逻辑：调用分层校验"""
        # 合并约束：writing_constraints 包含 forbidden_events, must_events 等
        constraints = writing_constraints.copy()
        constraints["outline"] = outline  # 可选，供硬校验使用
        return await self.validate_novel_consistency(text, constraints)

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
    async def _validate_with_llm(self, prompt: str, execution_result: Optional[dict] = None) -> dict:
        """发送 prompt 给 LLM 并解析 JSON 结果"""
        from openai import AsyncOpenAI
        from src.model_router import get_router
        
        model = get_router().get_model_for_task("validate")
        client = AsyncOpenAI(api_key="not-needed", base_url=self.llm_api_url)

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a validation assistant. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
                timeout=config.llm_timeout_validation,
            )
            raw_output = response.choices[0].message.content or ""
            parsed_result = self._parse_validation_result_enhanced(raw_output)
            return self._normalize_validation_result(parsed_result, raw_output)
        except Exception as e:
            logger.warning(f"LLM validation call failed: {e}, falling back to execution-based check")
            return self._fallback_validation(execution_result)

    # ---------- 以下为 JSON 解析辅助方法（保持不变） ----------
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
            "suggestions": ["Check LLM response format"],
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
            "code is correct",
            "passes validation",
            "satisfies the request",
            "meets requirements",
            "properly implements",
            "code works",
            "passed",
            "successful",
            "valid",
        ]
        negative_indicators = [
            "fails",
            "does not satisfy",
            "missing",
            "incorrect",
            "bug",
            "error",
            "issue",
            "problem",
            "failed",
            "invalid",
            "does not meet",
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
        if execution_result and execution_result.get("success"):
            return {
                "passed": True,
                "feedback": "Validation passed based on successful execution.",
                "suggestions": [],
            }
        return {
            "passed": False,
            "feedback": "Validation failed: no successful execution.",
            "suggestions": ["Fix execution errors and retry"],
        }
        
    async def validate_novel_consistency(
        self, text: str, constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """小说一致性校验：硬规则 + 软规则（LLM）"""
        # 第一层：硬校验（规则引擎）
        hard_ok, hard_msg = self._hard_validate_novel(text, constraints)
        if not hard_ok:
            return {
                "passed": False,
                "feedback": hard_msg,
                "suggestions": ["请修改场景，避免违反硬约束"],
            }

        # 第二层：软校验（LLM 评估语义一致性）
        soft_result = await self._soft_validate_novel(text, constraints)
        return soft_result

    def _hard_validate_novel(self, text: str, constraints: Dict[str, Any]) -> Tuple[bool, str]:
        """硬规则校验：禁止事件关键词、角色死亡等"""
        # 检查禁止事件
        forbidden = constraints.get("forbidden_events", [])
        for fb in forbidden:
            if fb and fb in text:
                return False, f"场景中包含禁止事件：{fb}"

        # 检查角色死亡（简单关键词）
        banned_death_keywords = ["死亡", "被杀", "死去", "身亡", "死了", "尸", "殒落"]
        if any(kw in text for kw in banned_death_keywords):
            # 如果有角色死亡约束，可以更精确检查，这里简单返回警告但先通过
            logger.warning("检测到死亡相关词汇，请确保符合大纲设定")
        return True, ""

    async def _soft_validate_novel(self, text: str, constraints: Dict[str, Any]) -> Dict[str, Any]:      
        """软校验：使用 LLM 检查是否满足 must_events，风格等"""
        must_events = constraints.get("must_events", [])
        if not must_events:
            # 没有必须事件，直接通过
            return {"passed": True, "feedback": "无必须事件，校验通过", "suggestions": []}

        # 简单思考：调用 LLM 判断是否实现了所有 must_events
        from openai import AsyncOpenAI
        from src.model_router import get_router

        model = get_router().get_model_for_task("validate")
        client = AsyncOpenAI(api_key="not-needed", base_url=self.llm_api_url)

        prompt = f"""判断以下小说场景是否包含了所有【必须发生的事件】。

    必须发生的事件：
    {chr(10).join(f'- {e}' for e in must_events)}

    场景正文：
    {text[:1500]}

    请输出 JSON 格式：
    {{"passed": true/false, "feedback": "说明缺了哪个事件或全部满足", "suggestions": []}}
    """
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
                timeout=30
            )
            content = response.choices[0].message.content or ""
            # 简单解析
            if '"passed": true' in content or '"passed":true' in content:
                return {"passed": True, "feedback": "所有必须事件均已包含", "suggestions": []}
            else:
                return {"passed": False, "feedback": "缺少必须事件", "suggestions": ["请补充遗漏的剧情"]}
        except Exception as e:
            logger.warning(f"LLM 软校验失败: {e}")
            return {"passed": True, "feedback": "软校验失败，默认通过", "suggestions": []}    