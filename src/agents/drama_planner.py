# src/agents/drama_planner.py
"""
Drama Planner - 将物流规划转化为戏剧结构

职责：
- 接收 Planner 的场景计划（goal, conflict, outcome, characters, must_events）
- 输出结构化的戏剧指令（goal, obstacle, pressure, decision, cost, relationship_delta）
- 确保 Writer 接收到完整的"欲望→阻碍→压力→选择→代价"链条
"""

import json
import re
import time
import httpx
from typing import Dict, Any, Optional
from openai import AsyncOpenAI

from src.agents.base import BaseAgent
from src.orchestrator.state import AgentState
from src.common.logging import setup_logging
from src.execution.llm_router_pool import get_llm_router_pool
from src.model_router import get_router
from src.common.prompt_logger import log_prompt
from src.config import config

logger = setup_logging("agents.drama_planner")

DRAMA_PLANNER_SYSTEM_PROMPT = """你是一位戏剧结构规划师（Drama Planner）。你的唯一职责是：**让场景变得困难**。

你接收 Planner 的场景计划，输出结构化的戏剧指令，确保 Writer 生成的内容包含完整的戏剧链条：

**欲望 → 阻碍 → 压力 → 选择 → 代价**

## 核心原则

1. **每个场景必须有明确的阻碍**：有人或事物阻止主角获得想要的东西
2. **每个场景必须有压力**：时间限制、资源不足、道德困境
3. **主角必须做选择**：至少面临一次两难抉择
4. **选择必须有代价**：无论选什么，都会失去或承担后果
5. **关系必须有变化**：至少一个角色的关系发生改变

## 输出格式

你必须输出严格的 JSON，包含以下字段：

```json
{
  "scene_goal": "主角在场景中想要什么（一句话）",
  "obstacle": {
    "type": "authority|enemy|environment|internal",
    "description": "具体的阻碍描述"
  },
  "pressure": {
    "type": "time_limit|resource_limit|moral_dilemma",
    "description": "具体的压力描述"
  },
  "decision": {
    "options": ["选项1", "选项2"],
    "chosen": "主角最终选择（必须从 options 中选择）"
  },
  "cost": {
    "success": "如果成功，主角失去什么",
    "failure": "如果失败，主角失去什么"
  },
  "relationship_delta": {
    "target": "关系对象",
    "from": "原有关系状态",
    "to": "新的关系状态"
  },
  "scene_role": "ESCALATION"
}

关键要求

    obstacle 必须有具体的人或事物，不能是抽象的"困难"

    pressure 必须有明确的紧迫感（时间、资源、道德）

    decision 的两个选项必须都有合理的代价

    cost 必须是具体的损失，而非模糊的"后果"

    relationship_delta 必须涉及场景中已有的角色

记住：你不是在写小说，你是在制造障碍。你的输出越具体、越困难，Writer 生成的故事就越有张力。
"""

class DramaPlannerAgent(BaseAgent):
    """戏剧结构规划师 - 将场景计划转化为戏剧结构"""

    def __init__(self):
        """初始化 Drama Planner"""
        logger.info("DramaPlannerAgent initialized")

    async def run(self, state: AgentState) -> Dict[str, Any]:
        """
        执行 Drama Planner，生成戏剧结构
        
        Args:
            state: AgentState，包含 scene_plan
            
        Returns:
            包含 drama_structure 的字典
        """
        start_time = time.time()
        logger.info("DramaPlannerAgent starting")

        scene_plan = state.scene_plan or {}
        if not scene_plan:
            logger.warning("DramaPlannerAgent: no scene_plan, returning empty")
            return {"drama_structure": {}}

        # 1. 构建 Prompt
        prompt = self._build_prompt(scene_plan)

        # 2. 记录 prompt 日志
        log_prompt("drama_planner", prompt, metadata={
            "novel_id": state.novel_id,
            "chapter": state.current_chapter,
            "scene_idx": state.current_scene_index
        })

        # 3. 调用 LLM
        raw_output = await self._call_llm(prompt)

        # 4. 解析输出
        drama_structure = self._parse_output(raw_output)

        # 5. 验证结构完整性
        if not self._validate_structure(drama_structure):
            logger.warning("DramaPlannerAgent: invalid structure, using fallback")
            drama_structure = self._build_fallback_structure(scene_plan)

        duration = time.time() - start_time
        logger.info(f"DramaPlannerAgent completed in {duration:.2f}s")

        return {
            "drama_structure": drama_structure,
            "metadata": state.metadata,
        }
        
    def _build_prompt(self, scene_plan: Dict) -> str:
        """构建 Drama Planner 的 Prompt"""
        goal = scene_plan.get("goal", "")
        conflict = scene_plan.get("conflict", "")
        outcome = scene_plan.get("outcome", "")
        characters = scene_plan.get("characters", [])
        must_events = scene_plan.get("must_events", [])

        return f"""根据以下场景计划，生成戏剧结构。

**场景计划**：
- 目标：{goal}
- 冲突：{conflict}
- 结果：{outcome}
- 角色：{', '.join(characters) if characters else '未指定'}
- 必须事件：{', '.join(must_events) if must_events else '无'}

请按照系统指令，为这个场景设计戏剧结构。确保包含：
1. 明确的阻碍（obstacle）
2. 明确的压力（pressure）
3. 两难选择（decision）
4. 具体的代价（cost）
5. 关系变化（relationship_delta）

只输出 JSON，不要有任何额外文本。"""

    async def _call_llm(self, prompt: str) -> str:
        router = get_router()
        model = router.get_model_for_task("plan")
        pool = get_llm_router_pool()

        async def _do_call(model_name: str, **kwargs) -> str:
            base_url = pool.get_base_url(model_name)
            timeout = httpx.Timeout(600.0, connect=30.0)  # 增加到 600 秒
            client = AsyncOpenAI(api_key="not-needed", base_url=base_url, timeout=timeout)
            resp = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": DRAMA_PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=1024,   # 限制输出长度，加速响应
            )
            return resp.choices[0].message.content or ""

        try:
            return await pool.call(model, _do_call, timeout=600, agent="drama_planner")
        except Exception as e:
            logger.error(f"Drama Planner LLM call failed: {e}, returning fallback")
            # 返回一个最小可用结构
            return '{"scene_goal": "推进场景", "obstacle": {"type": "unknown", "description": "遇到阻碍"}, "pressure": {"type": "time", "description": "时间紧迫"}, "decision": {"options": ["行动", "等待"], "chosen": "行动"}, "cost": {"success": "消耗资源", "failure": "错失机会"}, "relationship_delta": {"target": "主角", "from": "未知", "to": "坚定"}}'
        
    def _parse_output(self, raw: str) -> Dict[str, Any]:
        """从 LLM 输出中提取 JSON"""
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            logger.error("No JSON found in drama planner output")
            return {}

        try:
            data = json.loads(match.group())
            # 确保必要字段存在
            required = ["scene_goal", "obstacle", "pressure", "decision", "cost"]
            for field in required:
                if field not in data:
                    data[field] = {}
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse drama planner JSON: {e}")
            return {}

    def _validate_structure(self, structure: Dict[str, Any]) -> bool:
        """验证戏剧结构是否完整"""
        if not structure:
            return False

        required = ["scene_goal", "obstacle", "pressure", "decision", "cost"]
        for field in required:
            if field not in structure or not structure[field]:
                return False

        # 检查 obstacle 是否有 description
        if "description" not in structure.get("obstacle", {}):
            return False

        # 检查 decision 是否有 options 和 chosen
        decision = structure.get("decision", {})
        if not decision.get("options") or not decision.get("chosen"):
            return False

        # 检查 chosen 是否在 options 中
        if decision.get("chosen") not in decision.get("options", []):
            return False

        # 检查 cost 是否有 success 或 failure
        cost = structure.get("cost", {})
        if not cost.get("success") and not cost.get("failure"):
            return False

        return True

    def _build_fallback_structure(self, scene_plan: Dict) -> Dict[str, Any]:
        """当 LLM 解析失败时，构建一个最小可用的结构"""
        goal = scene_plan.get("goal", "推进剧情")
        characters = scene_plan.get("characters", ["主角"])
        target = characters[0] if characters else "主角"

        return {
            "scene_goal": f"{target}想要{goal}",
            "obstacle": {
                "type": "environment",
                "description": f"在完成{goal}的过程中遇到了意想不到的困难"
            },
            "pressure": {
                "type": "time_limit",
                "description": "时间紧迫，必须尽快完成"
            },
            "decision": {
                "options": ["谨慎行事", "冒险一搏"],
                "chosen": "冒险一搏"
            },
            "cost": {
                "success": "消耗了宝贵的资源",
                "failure": "失去了这次机会"
            },
            "relationship_delta": {
                "target": target,
                "from": "未定",
                "to": "坚定"
            }
        }
