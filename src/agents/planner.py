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
        user_request = state.original_request or state.user_input
        logger.info(f"Planning task: {user_request[:100]}...")
        try:
            response = await self._call_llm_with_fallback(user_request)
            plan = self._parse_response(response)
            plan.original_request = user_request
            plan_dict = plan.dict()
            execution_order = self._topological_sort(plan.subtasks)
            plan_dict["execution_order"] = execution_order
            duration = time.time() - start_time
            logger.info(f"{agent_name} completed, step={step}, status=success, duration={duration:.2f}")
            return {
                "task_plan": plan_dict,
                "plan_id": plan.plan_id,
                "subtasks": [st.description for st in plan.subtasks],
                "original_request": user_request,
            }
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{agent_name} failed, step={step}, error={e}, duration={duration:.2f}")
            logger.warning("Using fallback plan with single subtask")
            fallback_subtask = Subtask(
                id="task_1",
                name="Execute request",
                description=user_request,
                type="code",
                dependencies=[]
            )
            fallback_plan = TaskPlan(
                plan_id="fallback",
                original_request=user_request,
                subtasks=[fallback_subtask]
            )
            plan_dict = fallback_plan.dict()
            plan_dict["execution_order"] = ["task_1"]
            return {
                "task_plan": plan_dict,
                "plan_id": "fallback",
                "subtasks": [user_request],
            }

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