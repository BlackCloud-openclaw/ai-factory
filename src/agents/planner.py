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
from src.prompts.planner_prompts import PROMPT_REGISTRY
from src.writing.causality.rule_engine import RuleEngine
from src.common.prompt_logger import log_prompt

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

# 用于普通代码/任务规划的 prompt（保留原样）
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
        self.rule_engine = None

    def _get_rule_engine(self):
        if self.rule_engine is None:
            from src.writing.causality.rule_engine import RuleEngine
            self.rule_engine = RuleEngine()
        return self.rule_engine

    async def run(self, state: AgentState) -> Dict[str, Any]:
        self._state = state  # 保存供日志使用
        
        agent_name = "PlannerAgent"
        state.step_count += 1
        step = state.step_count
        logger.info(f"Starting {agent_name}, step={step}")
        start_time = time.time()

        task_type = getattr(state, 'task_type', 'code')

        # ---------- 分步式大纲生成（长篇小说专用） ----------
        if task_type == "novel_outline":
            try:
                outline = await self._generate_outline_step_by_step(state)
                duration = time.time() - start_time
                logger.info(f"{agent_name} completed (stepwise outline), step={step}, status=success, duration={duration:.2f}")
                return {
                    "plan_result": outline,
                    "outline": outline,
                    "error": None,
                }
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{agent_name} failed (stepwise outline), step={step}, error={e}, duration={duration:.2f}")
                return {"plan_result": {}, "error": str(e)}

        # ---------- 对于 scene_plan 任务，计算可供性提示（冷却） ----------
        if task_type == "scene_plan":
            # 获取所有有 enables 的规则（即可供性规则）
            affordance_rules = [r for r in self._get_rule_engine().rules if r.enables]
            if affordance_rules and state.novel_id:
                from src.writing.causality.affordance import get_affordance_cooldown_penalty
                scored = []
                for rule in affordance_rules:
                    # 使用第一个能力作为标识（可根据规则定义调整）
                    aff_id = rule.enables[0] if rule.enables else rule.id
                    penalty = await get_affordance_cooldown_penalty(
                        state.novel_id, aff_id, state.current_chapter, rule.cooldown
                    )
                    # 基础分数（此处简单使用 1，未来可结合谓词满足程度）
                    score = 1.0 * penalty
                    hint_text = rule.hint if rule.hint else rule.suggestion   # 关键修改
                    scored.append((score, rule.hint))
                # 按分数降序排序，取前 5
                scored.sort(reverse=True, key=lambda x: x[0])
                top_hints = [hint for _, hint in scored[:5]]
                state.metadata["affordance_hints"] = top_hints
            else:
                state.metadata["affordance_hints"] = []

        # ---------- 其他任务类型（code, scene_plan 等）使用原有逻辑 ----------
        builder = PROMPT_REGISTRY.get(task_type)
        if not builder:
            builder = PROMPT_REGISTRY["code"]

        prompt = builder.build(state)   # builder.build 内部会读取 state.metadata 中的 affordance_hints

        try:
            response = await self._plan_request_with_prompt(prompt, task_type)
            result = builder.parse_response(response)
            logger.debug(f"Raw LLM response for task_type={task_type}: {response[:500]}")

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

    # ========== 分步式大纲生成核心逻辑 ==========
    async def _generate_outline_step_by_step(self, state: AgentState) -> Dict[str, Any]:
        user_input = state.user_input

        # 1. 解析用户需求中的卷数和每卷章数（简单正则，可扩展）
        volumes_match = re.search(r'(\d+)\s*卷', user_input)
        total_volumes = int(volumes_match.group(1)) if volumes_match else 5
        chapters_per_vol_match = re.search(r'每卷\s*(\d+)\s*章', user_input)
        chapters_per_vol = int(chapters_per_vol_match.group(1)) if chapters_per_vol_match else 10
        logger.info(f"Stepwise outline: total_volumes={total_volumes}, chapters_per_vol={chapters_per_vol}")

        # 2. 生成卷列表（使用 volumes_outline builder）
        volumes_builder = PROMPT_REGISTRY.get("volumes_outline")
        if not volumes_builder:
            raise ValueError("Missing 'volumes_outline' builder in PROMPT_REGISTRY")
        prompt_vol = volumes_builder.build(state)
        resp_vol = await self._plan_request_with_prompt(prompt_vol, "volumes_outline")
        vol_result = volumes_builder.parse_response(resp_vol)
        volumes = vol_result.get("volumes", [])
        # 补足缺失的卷数（防止模型输出少于预期）
        while len(volumes) < total_volumes:
            new_vol = {
                "volume_num": len(volumes)+1,
                "title": f"第{len(volumes)+1}卷",
                "target_realm": "元婴",
                "core_conflict": "继续冒险提升实力"
            }
            volumes.append(new_vol)
            logger.warning(f"Auto-filled missing volume {len(volumes)}")

        full_outline = {
            "title": "修仙长路",
            "world_rules": ["灵力等级体系", "弱肉强食规则"],
            "characters": [{"name": "林逸", "initial_state": {"realm": "炼气", "level": 1}}],
            "volumes": []
        }

        # 3. 逐卷生成章节列表
        chapters_builder = PROMPT_REGISTRY.get("chapters_outline")
        if not chapters_builder:
            raise ValueError("Missing 'chapters_outline' builder in PROMPT_REGISTRY")

        for vol in volumes:
            # 将当前卷信息临时放入 state.metadata，供 builder 使用
            state.metadata["current_volume_info"] = vol
            state.metadata["chapters_per_vol"] = chapters_per_vol
            prompt_ch = chapters_builder.build(state)
            resp_ch = await self._plan_request_with_prompt(prompt_ch, "chapters_outline")
            ch_result = chapters_builder.parse_response(resp_ch)
            chapters = ch_result.get("chapters", [])
            # 补足缺失的章节
            while len(chapters) < chapters_per_vol:
                new_ch = {
                    "chapter_num": len(chapters)+1,
                    "title": f"第{len(chapters)+1}章",
                    "must_events": [],
                    "forbidden_events": []
                }
                chapters.append(new_ch)
                logger.warning(f"Auto-filled missing chapter {len(chapters)} in volume {vol['volume_num']}")
            vol["chapters"] = chapters
            full_outline["volumes"].append(vol)

        logger.info(f"Stepwise outline generated: {len(full_outline['volumes'])} volumes, total chapters={sum(len(v['chapters']) for v in full_outline['volumes'])}")
        return full_outline

    # ========== 以下是原有的辅助方法，保持不变 ==========
    async def _plan_request_with_prompt(self, prompt: str, task_type: str) -> str:
        """直接发送自定义 prompt 给 LLM，返回原始响应。"""
        router = get_router()
        pool = get_llm_router_pool()

        if task_type in ("scene_plan", "novel_outline", "volumes_outline", "chapters_outline"):
            model_name = router.get_model_for_task("plan")
            candidates = [model_name]
        elif task_type == "code":
            model_name = router.get_model_for_task("code")
            candidates = [model_name]
        else:
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
                    max_tokens=8192
                )
                message = response.choices[0].message
                content = message.content or ""
                if not content and hasattr(message, "reasoning_content"):
                    content = message.reasoning_content or ""
                # ===== 添加日志 =====
                logger.info(f"Planner raw response (first 2000 chars):\n{content[:2000]}")
                logger.info(f"Planner response length: {len(content)}")
                return content
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                raise
            
        # 提取约束（用于日志）
        constraints = None
        if hasattr(self, '_state') and self._state:
            from src.writing.state_projection import extract_hard_constraints
            from src.writing.world_state import WorldState
            if self._state.current_state:
                world = WorldState.from_dict(self._state.current_state)
                constraints = extract_hard_constraints(world)
        
        # 记录 prompt
        log_prompt("planner", prompt, metadata={"task_type": task_type}, constraints=constraints)        
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
            max_tokens=8192
        )
        content = response.choices[0].message.content
        if not content:
            content = getattr(response.choices[0].message, "reasoning_content", None)
        if not content:
            raise ValueError("Empty response from LLM")
        return content

    def _parse_response(self, response: str) -> TaskPlan:
        response = response.strip()
        match = re.search(r'```json\s*([\s\S]*?)\s*```', response, re.DOTALL)
        if match:
            response = match.group(1).strip()
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
            end = len(response) - 1
            response = response[:end+1] + '}' * brace_count
        json_str = response[start:end+1]
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        data = json.loads(json_str)
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