# src/prompts/validator_prompts.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class ValidatorPromptBuilder(ABC):
    @abstractmethod
    def build(self, target: str, user_input: str, context: Optional[Dict] = None) -> str:
        pass

class CodeValidatorPromptBuilder(ValidatorPromptBuilder):
    def build(self, code: str, user_input: str, execution_result: Optional[Dict] = None) -> str:
        exec_status = "unknown"
        exec_output = ""
        if execution_result:
            if execution_result.get("success"):
                exec_status = "passed"
                exec_output = execution_result.get("stdout", "")[:500]
            else:
                exec_status = "failed"
                exec_output = execution_result.get("stderr", "")[:500]
        exec_output_section = f"Execution Output:\n{exec_output}" if exec_output else "No execution output available."

        return f"""You are a code quality validator. Your job is to verify whether generated code:

1. Correctly implements the user's requirements (focus on main functionality)
2. Is syntactically and logically correct (but for tool modules, if it executes successfully, consider it logically correct)
3. Handles edge cases appropriately (only basic checks)
4. Follows Python best practices (basic only)

Return your evaluation in **strict JSON format** with these exact keys:
{{
    "passed": true/false,
    "feedback": "detailed explanation",
    "suggestions": ["improvement1", "improvement2"]
}}

== AI Factory ToolsRegistry Specification (MUST follow when the task is about creating a tool) ==

If the user request asks for a tool that can be registered into AI Factory ToolsRegistry, the **only** criteria for passing are:

- The code MUST include a function `get_tool_info()` returning a dict with keys: "name", "description", "module_path", "function_name", "parameters".
- The module MUST implement the function named by `function_name` (the main function).
- The code MUST NOT contain `if __name__ == "__main__":` block.
- The code MUST NOT contain any test code (unittest, pytest, manual test calls).
- The code MUST NOT define any class (only functions).
- The code MUST NOT use decorators (e.g., `@register_tool`).
- The code MUST NOT create a custom registry (e.g., `tools_registry = {{}}`).
- **Imports are allowed ONLY from Python standard library** (e.g., `urllib`, `json`, `re`, etc.). Reject imports like `requests`, `bs4`, `scrapy`, `aiohttp`, `httpx`.

**Important**: Do NOT reject code for minor logical issues like exception handling details, as long as the code executes successfully and the main functionality works. If execution result shows success (no errors), you should generally pass the validation.

If the user request is **not** about tool creation, ignore the above tool-specific rules and focus on functional correctness and code quality.

You must consider execution results if provided. If the code executed successfully and no obvious fatal errors exist, it likely passes.

Be strict about missing required functions, prohibited patterns (classes, test code, etc.), and non-standard library imports. Be lenient on code style and minor error handling improvements.

{exec_output_section}

Return ONLY valid JSON, no extra text."""

class NovelValidatorPromptBuilder(ValidatorPromptBuilder):
    def build(self, scene_text: str, user_input: str, constraints: Optional[Dict] = None) -> str:
        if constraints is None:
            constraints = {}
        return f"""你是一个小说一致性校验专家。请检查以下生成的场景文本是否违反了给定的写前约束。

用户要求: {user_input}

写前约束：
- 必须发生的事件: {constraints.get('must_events', [])}
- 禁止事件: {constraints.get('forbidden_events', [])}
- 当前角色状态: {constraints.get('character_states', {{}})}
- 风格要求: {constraints.get('style_profile', {{}})}

生成的场景文本：
{scene_text}

请返回 JSON 格式评估：
{{
    "passed": true/false,
    "feedback": "详细说明通过或失败的原因",
    "suggestions": ["改进建议1", "改进建议2"]
}}"""

VALIDATOR_PROMPT_REGISTRY = {
    "code": CodeValidatorPromptBuilder(),
    "novel": NovelValidatorPromptBuilder(),
}