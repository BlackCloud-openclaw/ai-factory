# src/agents/planner.py
import re
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from src.config import config
from src.common.logging import setup_logging
from src.agents.base import BaseAgent
from src.orchestrator.state import AgentState
from src.model_router import get_router
from src.execution.llm_router_pool import get_llm_router_pool
from src.config import config
from src.prompts.planner_prompts import PROMPT_REGISTRY

logger = setup_logging("agents.planner")

class Subtask(BaseModel):
    id: str
    name: str
    description: str
    type: str = "code"
    dependencies: List[str] = Field(default_factory=list)
    required_tools: List[str] = Field(default_factory=list)

class TaskPlan(BaseModel):
    plan_id: str
    original_request: str
    subtasks: List[Subtask]
    created_at: datetime = Field(default_factory=datetime.now)

# 注意：所有 JSON 示例中的大括号必须转义为双括号，只有 {user_request} 是真正的占位符
PLANNER_PROMPT = """You are a Task Planner. Break down the user request into a sequence of subtasks (each with a type, description, and dependencies). Use only these types: code, research, validate.

Return ONLY a valid JSON object with this structure:
{{
    "plan_id": "unique_id",
    "subtasks": [
        {{
            "id": "task_1",
            "name": "short name",
            "description": "detailed instruction",
            "type": "code",
            "dependencies": []
        }},
        {{
            "id": "task_2",
            "name": "another task",
            "description": "...",
            "type": "code",
            "dependencies": ["task_1"]
        }}
    ]
}}

Special instruction for tool creation tasks:
If the user request is about creating a tool that can be registered into AI Factory ToolsRegistry (keywords: 工具、tool、注册、ToolsRegistry), the validate subtask MUST NOT ask for testing code. Instead, the description should be: "验证代码是否符合 ToolsRegistry 规范：包含 get_tool_info() 和主函数，无测试代码、无类、无装饰器、无非标准库导入。"

User request: {user_request}
"""

class PlannerAgent(BaseAgent):
    def __init__(self):
        pass
                
    async def run(self, state: AgentState) -> Dict[str, Any]:
        agent_name = "PlannerAgent"
        state.step_count += 1
        step = state.step_count
        logger.info(f"Starting {agent_name}, step={step}")
        start_time = time.time()

        task_type = getattr(state, 'task_type', 'code')
        builder = PROMPT_REGISTRY.get(task_type)
        if not builder:
            builder = PROMPT_REGISTRY["code"]   # fallback

        prompt = builder.build(state)

        try:
            response = await self._plan_request_with_prompt(prompt, task_type)
            result = builder.parse_response(response)
            logger.debug(f"Raw LLM response for task_type={task_type}: {response[:500]}")

            # 如果 scene_plan 是列表，包装为字典
            if task_type == "scene_plan" and isinstance(result, list):
                result = {"scenes": result}

            duration = time.time() - start_time
            logger.info(f"{agent_name} completed, step={step}, status=success, duration={duration:.2f}")
            return {
                "plan_result": result,
                "task_plan": result if task_type == "code" else None,
                "outline": result if task_type == "novel_outline" else None,
                "scene_plan": result if task_type == "scene_plan" else None,
                "error": None,
            }
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{agent_name} failed, step={step}, error={e}, duration={duration:.2f}")
            return {"plan_result": {}, "error": str(e)}

   
    async def _plan_request_with_prompt(self, prompt: str, task_type: str) -> str:
        """直接发送自定义 prompt 给 LLM，返回原始响应。
        
        Args:
            prompt: 构造好的提示词
            task_type: 任务类型 (code, novel_outline, scene_plan等)
        """
        from src.model_router import get_router
        from src.execution.llm_router_pool import get_llm_router_pool
        from src.config import config

        router = get_router()
        pool = get_llm_router_pool()

        # 根据任务类型确定使用的模型
        if task_type in ("scene_plan", "novel_outline"):
            # 计划和提纲生成使用 plan 模型
            model_name = router.get_model_for_task("plan")
            candidates = [model_name]
        elif task_type == "code":
            model_name = router.get_model_for_task("code")
            candidates = [model_name]
        else:
            # 降级：使用默认模型
            model_name = router.get_model_for_task("default")
            candidates = [model_name]

        async def _call_llm(model, *args, **kwargs):
            logger.info(f"Planner using model: {model}")
            from openai import AsyncOpenAI
            api_url = kwargs.get('base_url', config.llm_api_url)
            client = AsyncOpenAI(api_key="not-needed", base_url=api_url)
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=2048
                )
                message = response.choices[0].message
                content = message.content or ""
                if not content and hasattr(message, "reasoning_content"):
                    content = message.reasoning_content or ""
                logger.info(f"LLM response length: {len(content)}")
                return content
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                raise

        return await pool.call_with_fallback(candidates, _call_llm, timeout=config.llm_timeout_planning)
    

    async def _call_llm_with_fallback(self, user_request: str) -> str:
        router = get_router()
        candidates = router.get_candidates(user_request)
        pool = get_llm_router_pool()
        try:
            return await pool.call_with_fallback(
                candidates,
                self._plan_request,
                user_request,
                timeout=config.llm_timeout_planning
            )
        except Exception as e:
            logger.error(f"All candidate models failed for planning: {e}")
            raise

    async def _plan_request(self, model: str, user_request: str, base_url: Optional[str] = None) -> str:
        from openai import AsyncOpenAI
        api_url = base_url or config.llm_api_url
        client = AsyncOpenAI(api_key="not-needed", base_url=api_url)
        prompt = PLANNER_PROMPT.format(user_request=user_request)
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs only JSON. No extra text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2048
        )
        content = response.choices[0].message.content
        if not content:
            content = getattr(response.choices[0].message, "reasoning_content", None)
        if not content:
            raise ValueError("Empty response from LLM")
        return content

    def _parse_response(self, response: str) -> TaskPlan:
        # 原有思路基础上，增加自动补全括号的逻辑
        response = response.strip()
        # 提取代码块
        match = re.search(r'```json\s*([\s\S]*?)\s*```', response, re.DOTALL)
        if match:
            response = match.group(1).strip()
        # 找到第一个 { 和匹配的 }，如果到最后都不匹配则补全
        start = response.find('{')
        if start == -1:
            raise ValueError("No JSON object found")
        brace_count = 0
        end = start
        for i, ch in enumerate(response[start:], start):
            if ch == '{':
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i
                    break
        else:
            # 未找到匹配的结束括号，尝试补全
            end = len(response) - 1
            # 补全缺失的 } 并继续解析
            response = response[:end+1] + '}' * brace_count
        json_str = response[start:end+1]
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        data = json.loads(json_str)
        # ... 后续不变
        subtasks = []
        for st in data.get("subtasks", []):
            subtasks.append(Subtask(
                id=st["id"],
                name=st.get("name", ""),
                description=st.get("description", ""),
                type=st.get("type", "code"),
                dependencies=st.get("dependencies", [])
            ))
        return TaskPlan(
            plan_id=data.get("plan_id", "plan_001"),
            original_request="",
            subtasks=subtasks
        )

    def _topological_sort(self, subtasks: List[Subtask]) -> List[str]:
        id_map = {st.id: st for st in subtasks}
        indeg = {st.id: 0 for st in subtasks}
        graph = {st.id: [] for st in subtasks}
        for st in subtasks:
            for dep in st.dependencies:
                if dep in id_map:
                    graph[dep].append(st.id)
                    indeg[st.id] += 1
        queue = [sid for sid, deg in indeg.items() if deg == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for nei in graph[node]:
                indeg[nei] -= 1
                if indeg[nei] == 0:
                    queue.append(nei)
        if len(order) != len(subtasks):
            return [st.id for st in subtasks]
        return order