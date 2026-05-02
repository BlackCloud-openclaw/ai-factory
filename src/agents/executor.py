# src/agents/executor.py
import uuid
import time
import os
import re
from typing import Any, Dict, List, Optional
from functools import partial

from src.execution.sandbox import CodeSandbox
from src.execution.file_ops import FileOperations
from src.execution.llm_router_pool import get_llm_router_pool
from src.config import config
from src.common.logging import setup_logging
from src.orchestrator.state import AgentState
from src.agents.base import BaseAgent
from src.model_router import get_router
from src.config import config

logger = setup_logging("agents.executor")

CODER_SYSTEM_PROMPT = """
你是一名专业的 Python 程序员。请根据任务描述生成干净、带良好文档和测试的 Python 代码。

## 通用规则（始终遵守）
1. 只输出有效的 Python 代码，不要添加任何解释文字。
2. 包含 docstring 和必要的注释。
3. 处理边界情况和错误（如使用异常）。
4. 适当使用类型提示（type hints）。
5. 代码必须放在 Markdown 代码块中，并注明语言标签：```python ... ```

## 工具注册规范（当任务要求生成可供 AI Factory ToolsRegistry 自动注册的工具模块时，必须严格遵守）
如果任务描述中明确要求“生成一个可以自动注册到 AI Factory ToolsRegistry 的工具”，你必须生成一个完全符合以下规范的 Python 模块：

### 必须实现的函数：`get_tool_info()`
- 返回一个字典，包含以下键：
  - `"name"`: `str`，工具的唯一名称（例如 `"calculator"`）
  - `"description"`: `str`，简短描述（若用户输入为中文则使用中文）
  - `"module_path"`: `str`，通常设为空字符串（由系统自动填充）
  - `"function_name"`: `str`，工具主函数的名称（例如 `"calculate"`）
  - `"parameters"`: `dict`，描述每个参数的类型和约束，格式如下例

**示例**：
```python
def get_tool_info():
    return {
        "name": "calculator",
        "description": "提供加、减、乘、除四则运算",
        "module_path": "",
        "function_name": "calculate",
        "parameters": {
            "operation": {"type": "string", "enum": ["add", "sub", "mul", "div"], "description": "运算类型"},
            "a": {"type": "number", "description": "第一个数"},
            "b": {"type": "number", "description": "第二个数"}
        }
    }

### 必须实现的主函数
- 函数名由 get_tool_info() 中的 "function_name" 指定（如上例中的 calculate）。
- 必须接受 parameters 中声明的所有参数。
- 执行请求的操作，并返回结果（通常是 float 或 int）。
- 对于无效输入（如被零除、类型错误），应抛出合适的异常（如 ZeroDivisionError, ValueError）。
- 包含清晰的 docstring（若用户输入为中文则使用中文）。
- 主函数必须直接实现工具逻辑，不要包装或装饰。

### 禁止内容
- 不要包含 if __name__ == "__main__": 代码块。
- 不要包含任何测试代码或 unittest 类。
- 不要定义与工具无关的额外类或函数。
- 不要自行实现注册机制，只需提供上述要求的函数。
- 禁止定义任何装饰器（如 `@register_tool`）。  
- 禁止创建全局注册表（如 `tools_registry = {}`）。  
- 禁止定义任何类（如 `class ToolsRegistry`）。  
- 禁止一个模块包含多个工具函数（只能有一个由 `function_name` 指定的主函数）。  
- 禁止包含示例工具、示例调用、测试代码或 `if __name__ == "__main__"` 块。  
- 只允许使用 `urllib.request`, `json`, `re`, `typing`, `math`, `datetime`, `collections` 等标准库。特别禁止 `requests`, `beautifulsoup4`, `scrapy` 等第三方库。

### 代码风格
- 函数名和变量名使用 snake_case。
- 当工具描述为中文时，注释也使用中文。
- 保持模块聚焦：一个文件只实现一个工具。

### 依赖限制
- 只使用 Python 标准库，以确保兼容性。

重要提醒：生成的模块将被 AI Factory 的 ToolsRegistry 导入并调用 get_tool_info()，主函数将被直接调用。请严格遵守上述规范，只输出代码块，不要输出任何额外解释。
"""

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
        previous_validation = context.get("previous_validation")  # 新增

        code = await self.generate_with_fallback(user_input, research_results, subtasks, previous_validation=previous_validation)

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
            "previous_validation": state.validation_result,   # 新增
            "retry_count": state.retry_count,                 # 可选
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
        previous_validation: Optional[Dict[str, Any]] = None,   # 新增
    ) -> str:
        router = get_router()
        code_candidates = router.get_candidates(user_input)   # 正常获取 code 列表
        pool = get_llm_router_pool()
            
        # 确保 previous_validation 是字典
        if previous_validation is None:
            previous_validation = {}    
            
        # 如果有上一次验证失败的信息，构造一个附加提示
        previous_error_prompt = ""
        if previous_validation and not previous_validation.get("passed", True):
            feedback = previous_validation.get("feedback", "")
            suggestions = previous_validation.get("suggestions", [])
            if feedback:
                previous_error_prompt = f"\n\n[上一次生成的代码验证失败，错误信息: {feedback}]"
                if suggestions:
                    previous_error_prompt += f"\n[改进建议: {'; '.join(suggestions)}]"
                previous_error_prompt += "\n请根据上述错误信息重新生成正确的代码。"    
                
        # 创建绑定了 extra_prompt 的函数
        bound_func = partial(self._call_llm_for_code, extra_prompt=previous_error_prompt)
        
        # 第一次尝试：code 模型列表
        try:
            return await pool.call_with_fallback(
                code_candidates,
                bound_func,
                user_input, research_results, subtasks,
                timeout=config.llm_timeout_coding,
            )
        except Exception as e:
            logger.warning(f"All code models failed: {e}, falling back to research models")
        
        # 第二次尝试：research 模型列表
        research_candidates = router.candidates.get("research", [])
        if research_candidates:
            try:
                return await pool.call_with_fallback(
                    research_candidates,
                    bound_func,  # 使用同一个绑定了 extra_prompt 的函数
                    user_input, research_results, subtasks,
                    timeout=config.llm_timeout_coding                
                )
            except Exception as e:
                logger.warning(f"Research models also failed: {e}")
        else:
            logger.warning("No research models available")
        
        return ""

    async def _call_llm_for_code(
        self,
        model: str,
        user_input: str,
        research_results: List[Dict[str, Any]],
        subtasks: List[str],
        base_url: Optional[str] = None,
        extra_prompt: str = "",   # 新增
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

Please generate Python code that fulfills the above task, using the research context where applicable.
{extra_prompt}"""
        else:
            prompt = f"{task_description}\n\nPlease generate Python code that fulfills the above task.\n{extra_prompt}"

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