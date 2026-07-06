# src/agents/director.py
import json
import re
import time
import httpx
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI

from src.agents.base import BaseAgent
from src.orchestrator.state import AgentState
from src.common.logging import setup_logging
from src.execution.llm_router_pool import get_llm_router_pool
from src.model_router import get_router
from src.common.prompt_logger import log_prompt
from src.writing.director_models import (
    NarrativeBlueprint, KnowledgeDelta, CharacterIntent, SceneRole
)
from src.writing.world_state import WorldState
from src.config import config
from src.writing.character_intent_memory import CharacterIntentMemory
from src.prompts.director_prompts import DIRECTOR_ENHANCED_PROMPT as DIRECTOR_SYSTEM_PROMPT

logger = setup_logging("agents.director")


class DirectorAgent(BaseAgent):
    async def run(self, state: AgentState) -> Dict[str, Any]:
        start_time = time.time()
        logger.info("DirectorAgent starting")

        scene_plan = state.scene_plan or {}
        world_state = WorldState.from_dict(state.current_state) if state.current_state else WorldState()

        # 1. 加载意图记忆
        compressed_state = getattr(state, 'compressed_state', None) or state.metadata.get("compressed_state")
        intent_memory = CharacterIntentMemory(state.novel_id, compressed_state)

        # 2. 构建 prompt（不含意图记忆）
        prompt = self._build_prompt(scene_plan, world_state)

        # 3. 添加已有意图作为上下文
        existing_intents = intent_memory.get_all_intents_prompt()
        if existing_intents:
            prompt += f"\n\n{existing_intents}\n"
        
        # 4. 添加角色认知身份上下文
        identity_context = self._build_identity_context(world_state, scene_plan.get("characters", []))
        if identity_context:
            prompt += f"\n\n{identity_context}\n"

        # 5. 记录 prompt 日志
        log_prompt("director", prompt, metadata={
            "novel_id": state.novel_id,
            "chapter": state.current_chapter,
            "scene_idx": state.current_scene_index
        })

        # 6. 调用 LLM
        raw_output = await self._call_llm(prompt)

        # 7. 解析输出
        blueprint_data, deltas, intent_data = self._parse_output(raw_output)

        # 8. 构建对象
        blueprint = NarrativeBlueprint(
            attention_path=blueprint_data.get("attention_path", []),
            withheld_information=blueprint_data.get("withheld_information", ""),
            reveal_beat=blueprint_data.get("reveal_beat", ""),
            scene_pressure=blueprint_data.get("scene_pressure", ""),
            silent_action_priority=blueprint_data.get("silent_action_priority", ""),
            recurring_image=blueprint_data.get("recurring_image", ""),
            scene_role=SceneRole(blueprint_data.get("scene_role", "SETUP"))
        )

        knowledge_deltas = []
        for kd in deltas:
            knowledge_deltas.append(KnowledgeDelta(
                holder=kd["holder"],
                information=kd["information"],
                operation=kd["operation"],
                trigger=kd["trigger"],
                visibility=kd["visibility"]
            ))

        char_intent = None
        if intent_data:
            char_intent = CharacterIntent(
                actor=intent_data.get("actor", ""),
                conscious_goal=intent_data.get("conscious_goal", ""),
                hidden_need=intent_data.get("hidden_need", ""),
                fear=intent_data.get("fear", ""),
                misconception=intent_data.get("misconception"),
                immediate_tactic=intent_data.get("immediate_tactic", ""),
                perceived_relationships=intent_data.get("perceived_relationships"),
                # 认知身份字段
                beliefs=intent_data.get("beliefs"),
                attachments=intent_data.get("attachments"),
                self_image=intent_data.get("self_image"),
                moral_boundaries=intent_data.get("moral_boundaries"),
                identity_change_reason=intent_data.get("identity_change_reason"),
            )

            # 9. 更新意图记忆（也会包含认知身份）
            intent_memory.update_from_director(char_intent.__dict__)
            if "character_intents" not in state.metadata:
                state.metadata["character_intents"] = {}
            state.metadata["character_intents"].update(intent_memory.to_dict())

        duration = time.time() - start_time
        logger.info(f"DirectorAgent completed in {duration:.2f}s")

        return {
            "narrative_blueprint": blueprint.__dict__,
            "knowledge_deltas": [kd.__dict__ for kd in knowledge_deltas],
            "character_intent": char_intent.__dict__ if char_intent else None,
            "metadata": state.metadata,
        }

    def _build_prompt(self, scene_plan: Dict, world_state: WorldState) -> str:
        goal = scene_plan.get("goal", "")
        conflict = scene_plan.get("conflict", "")
        must_events = scene_plan.get("must_events", [])
        # 从 scene_plan 中提取角色名，用于提示 Director 使用正确的角色名
        characters = scene_plan.get("characters", [])
        character_list = ", ".join(characters) if characters else "根据场景推断"
        
        return f"""根据以下场景骨架，设计读者体验蓝图。

**场景骨架**：
- 目标: {goal}
- 冲突: {conflict}
- 必须事件: {must_events}
- 参与角色: {character_list}

**重要提醒**：
- character_intent 中的 actor 必须使用场景中实际存在的角色名（从上面的"参与角色"中选择）
- 如果场景中有主角，actor 必须是主角的名字
- 不要虚构不存在的角色名

**当前世界状态摘要**：
{self._summarize_world(world_state)}

请按照系统指令输出 JSON。只输出 JSON，不要有任何额外文本。
"""

    def _summarize_world(self, world_state: WorldState) -> str:
        """生成世界状态的简要摘要，用于 prompt"""
        lines = []
        protagonist = "林逸"
        if protagonist in world_state.characters:
            char = world_state.get_character(protagonist)
            lines.append(f"主角 {protagonist}：{char.full_realm()}，HP={char.hp}，位置={char.location}")
            if char.inventory:
                lines.append(f"背包：{', '.join(char.inventory[:5])}")
        if world_state.map.current:
            lines.append(f"当前位置：{world_state.map.current}")
        critical_flags = [k for k, v in world_state.global_flags.items() if v is True][:5]
        if critical_flags:
            lines.append(f"已触发的关键标记：{', '.join(critical_flags)}")
        return "\n".join(lines) if lines else "(无额外状态)"
    
    def _build_identity_context(self, world_state: WorldState, scene_characters: List[str]) -> str:
        """构建角色认知身份上下文"""
        if not scene_characters:
            return ""
        
        lines = ["【角色认知身份（必须尊重，不可随意改变）】"]
        has_identity = False
        
        for char_name in scene_characters:
            if char_name not in world_state.characters:
                continue
            char = world_state.get_character(char_name)
            identity_parts = []
            
            if char.self_image:
                identity_parts.append(f"自我认知：{char.self_image}")
                has_identity = True
            if char.beliefs:
                identity_parts.append(f"核心信念：{', '.join(char.beliefs[:5])}")
                has_identity = True
            if char.attachments:
                identity_parts.append(f"重要依恋：{', '.join(char.attachments[:5])}")
                has_identity = True
            if char.moral_boundaries:
                identity_parts.append(f"道德底线：{', '.join(char.moral_boundaries[:5])}")
                has_identity = True
            
            if identity_parts:
                lines.append(f"\n【{char_name}】")
                lines.extend(identity_parts)
        
        if has_identity:
            lines.append("\n⚠️ 生成 character_intent 时，必须与上述身份一致。如需改变信念或自我认知，请在 intent 中设置 identity_change_reason 字段。")
            return "\n".join(lines)
        return ""

    async def _call_llm(self, prompt: str) -> str:
        router = get_router()
        model = router.get_model_for_task("plan")
        pool = get_llm_router_pool()

        async def _do_call(model_name: str, **kwargs) -> str:
            base_url = pool.get_base_url(model_name)
            timeout = httpx.Timeout(7200.0, connect=60.0)
            client = AsyncOpenAI(api_key="not-needed", base_url=base_url, timeout=timeout)
            resp = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=2048,
            )
            return resp.choices[0].message.content or ""

        return await pool.call(model, _do_call, timeout=config.llm_timeout_director, agent="director")

    def _parse_output(self, raw: str):
        """从 LLM 输出中提取 JSON，返回 (blueprint_dict, knowledge_deltas_list, intent_dict)"""
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            logger.error("No JSON found in director output")
            return {}, [], {}
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse director JSON: {e}")
            return {}, [], {}

        blueprint = {k: v for k, v in data.items()
                     if k in ["attention_path", "withheld_information", "reveal_beat",
                              "scene_pressure", "silent_action_priority", "recurring_image", "scene_role"]}
        deltas = data.get("knowledge_deltas", [])
        intent = data.get("character_intent", {})
        return blueprint, deltas, intent