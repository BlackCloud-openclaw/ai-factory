# src/agents/planner.py
import re
import json
import time
import asyncio
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
            
from src.config import config
from src.common.logging import setup_logging
from src.agents.base import BaseAgent
from src.orchestrator.state import AgentState
from src.model_router import get_router
from src.execution.llm_router_pool import get_llm_router_pool
from src.prompts.planner_prompts import PROMPT_REGISTRY
from src.writing.causality.rule_engine import RuleEngine
from src.common.prompt_logger import log_prompt
from src.writing.narrative_entropy import EntropyController, EntropyReport, ControlAction
from src.db import get_db_pool
from src.writing.loop_store import LoopStore
from src.writing.planning_contract import create_contract_from_dict
from src.writing.narrative_intent import (
    NarrativeIntent,
    SceneRole,
    NarrativeConsequence,
    NarrativeCondition,
)
from src.writing.planning_contract import (
    PlanningContract, Observables, StateChange,
    Intent, Execution, ExecutionUnit, ContractMetadata,
    SceneSpecification, ContractEnrichment
)
from src.writing.planner_output import PlannerOutput
from src.writing.projection_context import ProjectionContext
from src.writing.narrative_projection import NarrativeProjection

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
        """
        PlannerAgent 主入口。
        根据 task_type 执行不同任务：
        - novel_outline: 分步式大纲生成
        - scene_plan: 场景计划 + NarrativeIntent 生成
        - code: 代码任务规划
        """
        logger.info("🔥🔥🔥 PLANNER RUN IS EXECUTING 🔥🔥🔥")
        self._state = state
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
                    "planner_outputs": [],  # 大纲生成不产生 planner_outputs
                    "error": None,
                }
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{agent_name} failed (stepwise outline), step={step}, error={e}, duration={duration:.2f}")
                return {"plan_result": {}, "outline": None, "planner_outputs": [], "error": str(e)}

        # ---------- 对于 scene_plan 任务，计算可供性提示（冷却） ----------
        if task_type == "scene_plan":
            # 获取所有有 enables 的规则（即可供性规则）
            affordance_rules = [r for r in self._get_rule_engine().rules if r.enables]
            if affordance_rules and state.novel_id:
                from src.writing.causality.affordance import get_affordance_cooldown_penalty
                scored = []
                for rule in affordance_rules:
                    aff_id = rule.enables[0] if rule.enables else rule.id
                    penalty = await get_affordance_cooldown_penalty(
                        state.novel_id, aff_id, state.current_chapter, rule.cooldown
                    )
                    score = 1.0 * penalty
                    hint_text = rule.hint if rule.hint else rule.suggestion
                    scored.append((score, hint_text))
                scored.sort(reverse=True, key=lambda x: x[0])
                top_hints = [hint for _, hint in scored[:5]]
                state.metadata["affordance_hints"] = top_hints
            else:
                state.metadata["affordance_hints"] = []

            # ========== 熵控制子系统 ==========
            entropy_report = None
            compressed_state = state.compressed_state
            if compressed_state:
                local_entropy = compressed_state.get("local_entropy", 0.0)
                arc_entropy = compressed_state.get("arc_entropy", 0.0)
                civ_entropy = compressed_state.get("civilization_entropy", 0.0)
                entropy_report = EntropyReport(
                    local=local_entropy,
                    arc=arc_entropy,
                    civilization=civ_entropy,
                    details=compressed_state.get("entropy_details", {})
                )
            else:
                entropy_report = EntropyReport()

            control_actions = EntropyController.regulate(entropy_report)
            state.metadata["entropy_control_actions"] = [action.to_dict() for action in control_actions]

            if control_actions:
                logger.info(f"Entropy control actions: {[a.type for a in control_actions]}")
            else:
                logger.debug("No entropy control actions needed")

            if compressed_state:
                narrative_entropy = compressed_state.get("narrative_entropy", 0.0)
                state.metadata["narrative_entropy_warning"] = narrative_entropy > 0.7
                state.metadata["narrative_entropy_value"] = narrative_entropy
            else:
                state.metadata["narrative_entropy_warning"] = False

            # 防止叙事扩散
            if compressed_state:
                character_arcs = compressed_state.get("character_arcs", {})
                unresolved_arcs = sum(1 for status in character_arcs.values() if status != "resolved")
                max_active_arcs = 5
                if unresolved_arcs > max_active_arcs:
                    forbid_arcs_action = ControlAction(
                        type="forbid_new_arcs",
                        params={"duration_chapters": 2, "reason": f"已有 {unresolved_arcs} 个未解决弧线，超过限制 {max_active_arcs}"}
                    )
                    existing_actions = state.metadata.get("entropy_control_actions", [])
                    if not any(a.get("type") == "forbid_new_arcs" for a in existing_actions):
                        existing_actions.append(forbid_arcs_action.to_dict())
                        state.metadata["entropy_control_actions"] = existing_actions
                        logger.info(f"Preventing diffusion: forced forbid_new_arcs due to {unresolved_arcs} unresolved arcs (limit {max_active_arcs})")

            # ========== 确保有激活的 Loop ==========
            if state.novel_id:
                pool = get_db_pool()
                if pool:
                    loop_store = LoopStore(pool)
                    active_loop = await loop_store.get_active_loop(state.novel_id)
                    if active_loop is None:
                        loop_desc = "主角提升实力，探索世界，推动主线剧情"
                        if state.outline and "volumes" in state.outline:
                            volumes = state.outline.get("volumes", [])
                            if volumes and state.current_volume <= len(volumes):
                                vol = volumes[state.current_volume - 1]
                                loop_desc = vol.get("core_conflict", loop_desc)
                        active_loop = await loop_store.create_loop(
                            state.novel_id,
                            title=f"主线推进 - 卷{state.current_volume}",
                            description=loop_desc
                        )
                        logger.info(f"✅ Created new active Loop for novel {state.novel_id}: {active_loop.title}")
                    state.metadata["active_loop"] = {
                        "id": str(active_loop.id),
                        "title": active_loop.title,
                        "description": active_loop.description,
                        "progress": active_loop.progress,
                    }

        # ---------- 构建 Prompt ----------
        builder = PROMPT_REGISTRY.get(task_type)
        if not builder:
            builder = PROMPT_REGISTRY["code"]

        prompt = builder.build(state)

        # ========== Phase 13.2: 注入 NarrativeProjection ==========
        if task_type == "scene_plan" and state.projection is not None:
            try:
                from src.writing.projection_context import ProjectionContext
                ctx = ProjectionContext.from_projection(state.projection)
                projection_text = ctx.to_prompt_text()
                prompt = prompt + "\n\n" + projection_text
                logger.info(
                    f"[PlannerAgent] Injected Projection (version {state.projection.version}) "
                    f"with {len(state.projection.unresolved_threads)} unresolved threads"
                )
            except Exception as e:
                logger.warning(f"[PlannerAgent] Failed to inject Projection: {e}")

        # 初始化 planner_outputs（用于 scene_plan 返回）
        planner_outputs = []

        try:
            response = await self.plan_request_with_prompt(prompt, task_type)
            result = builder.parse_response(response)
            logger.debug(f"Raw LLM response for task_type={task_type}: {response[:500]}")

            if task_type == "scene_plan" and isinstance(result, list):
                result = {"scenes": result}

            # ========== 生成 Planning Contract + NarrativeIntent ==========
            if task_type == "scene_plan" and result and "scenes" in result:
                scenes = result["scenes"]
                planner_outputs = []  # 重新初始化，确保清空

                for idx, scene in enumerate(scenes):
                    try:
                        if "scene_id" not in scene:
                            scene["scene_id"] = f"scene_{state.current_volume}_{state.current_chapter}_{idx}"
                        if "chapter" not in scene:
                            scene["chapter"] = state.current_chapter
                        if "scene_index" not in scene:
                            scene["scene_index"] = idx

                        # ========== 直接构造 PlanningContract，保留 observables ==========
                        obs_data = scene.get("observables", {})
                        
                        # 🔥 清洗 state_changes 中的数据（转换 to_minor_stage 字符串、处理 hp_change）
                        if "state_changes" in obs_data:
                            cleaned_changes = []
                            for sc in obs_data["state_changes"]:
                                # 处理 realm_change: to_minor_stage 字符串转整数
                                if sc.get("type") == "realm_change":
                                    stage = sc.get("to_minor_stage")
                                    if isinstance(stage, str):
                                        stage_map = {"初期": 1, "中期": 5, "后期": 9, "巅峰": 9}
                                        if stage in stage_map:
                                            sc["to_minor_stage"] = stage_map[stage]
                                        else:
                                            try:
                                                sc["to_minor_stage"] = int(stage)
                                            except ValueError:
                                                sc["to_minor_stage"] = 1
                                    elif stage is None:
                                        sc["to_minor_stage"] = 1
                                
                                # 处理 hp_change -> plot_flag（避免 Pydantic 验证错误）
                                if sc.get("type") == "hp_change":
                                    sc["type"] = "plot_flag"
                                    actor = sc.get("actor", "unknown")
                                    sc["name"] = f"hp_{actor}_changed"
                                    sc["value"] = True
                                    sc.pop("delta", None)
                                    sc.pop("actor", None)
                                    sc.pop("new_hp", None)
                                
                                cleaned_changes.append(sc)
                            obs_data["state_changes"] = cleaned_changes
                        
                        # 现在构建 StateChange 对象
                        state_changes = []
                        for sc in obs_data.get("state_changes", []):
                            # 如果没有 id，生成一个稳定的 id
                            if not sc.get("id"):
                                import hashlib
                                raw = f"{scene['scene_id']}|{sc.get('type', 'unknown')}|{sc.get('name', '')}|{sc.get('actor', '')}"
                                sc_id = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]
                                sc["id"] = sc_id
                            state_changes.append(StateChange(**sc))
                        
                        observables = Observables(state_changes=state_changes)

                        # 构建执行单元
                        execution_units = []
                        for i, event in enumerate(scene.get("must_events", [])):
                            execution_units.append(ExecutionUnit(
                                id=f"U{i+1}",
                                label="action",
                                description=event,
                                attributes={}
                            ))

                        # 构建 scene_spec（如果有）
                        scene_spec = None
                        if scene.get("scene_spec"):
                            scene_spec = SceneSpecification(**scene.get("scene_spec"))

                        contract = PlanningContract(
                            version="1.0",
                            scene_id=scene["scene_id"],
                            intent=Intent(
                                goal=scene.get("goal", ""),
                                conflict=scene.get("conflict", ""),
                                expected_outcome=scene.get("outcome", "")
                            ),
                            execution=Execution(units=execution_units),
                            observables=observables,
                            constraints=[],  # 暂时为空，可由 Normalizer 补全
                            metadata=ContractMetadata(
                                chapter=state.current_chapter,
                                scene_index=idx,
                                arc=None
                            ),
                            scene_spec=scene_spec,
                            enrichment=ContractEnrichment()
                        )

                        # 构建 contract_dict（用于场景计划）
                        contract_dict = {
                            "version": contract.version,
                            "scene_id": contract.scene_id,
                            "intent": contract.intent.model_dump(),
                            "execution": contract.execution.model_dump(),
                            "observables": observables.model_dump(),
                            "constraints": [],
                            "metadata": contract.metadata.model_dump(),
                            "scene_spec": contract.scene_spec.model_dump() if contract.scene_spec else None,
                            "enrichment": contract.enrichment.model_dump(),
                        }
                        scene["planning_contract"] = contract_dict

                        # ========== 构建 NarrativeIntent ==========
                        narrative_intent = self._build_narrative_intent_from_scene(
                            scene=scene,
                            scene_id=scene["scene_id"],
                            chapter=state.current_chapter,
                            scene_idx=idx,
                        )

                        planner_output = PlannerOutput(
                            narrative_intent=narrative_intent,
                            execution_contract=contract,
                        )
                        planner_outputs.append(planner_output.model_dump())

                        logger.info(f"✅ Generated Planning Contract for scene {scene['scene_id']}")
                    except Exception as e:
                        logger.warning(f"Failed to generate Planning Contract for scene {idx}: {e}")
                        scene["planning_contract"] = None                 


            duration = time.time() - start_time
            logger.info(f"{agent_name} completed, step={step}, status=success, duration={duration:.2f}")

            # 构建返回字典
            return_dict = {
                "plan_result": result,
                "task_plan": result if task_type == "code" else None,
                "outline": result if task_type == "novel_outline" else None,
                "scene_plan": result if task_type == "scene_plan" else None,
                "planner_outputs": planner_outputs,  # 关键添加
                "error": None,
            }
            
            # ========== 在这里插入上面的验证日志 ==========
            if task_type == "scene_plan" and "scene_plan" in return_dict:
                scene_plan_data = return_dict["scene_plan"]
                if scene_plan_data and isinstance(scene_plan_data, dict) and "scenes" in scene_plan_data:
                    scenes_in_return = scene_plan_data["scenes"]
                    if scenes_in_return:
                        logger.info(f"🔍 Returning scenes[0] keys: {list(scenes_in_return[0].keys())}")
                        logger.info(f"🔍 Returning scenes[0] planning_contract: {scenes_in_return[0].get('planning_contract', 'MISSING')}")
                    else:
                        logger.warning("🔍 Returning scenes is empty")
                else:
                    logger.warning("🔍 Returning scene_plan is not a dict or missing 'scenes'")

            
            return return_dict

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{agent_name} failed, step={step}, error={e}, duration={duration:.2f}")
            return {"plan_result": {}, "planner_outputs": [], "error": str(e)}

    # ========== 分步式大纲生成核心逻辑（支持分批生成章节） ==========
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
        resp_vol = await self.plan_request_with_prompt(prompt_vol, "volumes_outline")
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

        # 3. 逐卷生成章节列表（分批，每批10章）
        chapters_builder = PROMPT_REGISTRY.get("chapters_outline")
        if not chapters_builder:
            raise ValueError("Missing 'chapters_outline' builder in PROMPT_REGISTRY")

        for vol in volumes:
            # 将当前卷信息临时放入 state.metadata，供 builder 使用
            state.metadata["current_volume_info"] = vol
            state.metadata["chapters_per_vol"] = chapters_per_vol
            
            all_chapters = []
            batch_size = 10   # 每批生成10章，避免输出过长
            total_needed = chapters_per_vol
            
            while len(all_chapters) < total_needed:
                start_num = len(all_chapters) + 1
                end_num = min(start_num + batch_size - 1, total_needed)
                state.metadata["chapter_range"] = (start_num, end_num)
                
                prompt_ch = chapters_builder.build(state)
                # 为章节生成设置更大的 max_tokens（在 plan_request_with_prompt 中处理）
                resp_ch = await self.plan_request_with_prompt(prompt_ch, "chapters_outline")
                ch_result = chapters_builder.parse_response(resp_ch)
                batch_chapters = ch_result.get("chapters", [])
                
                expected = end_num - start_num + 1
                if len(batch_chapters) < expected:
                    logger.warning(f"Batch for volume {vol['volume_num']} returned {len(batch_chapters)} chapters, expected {expected}. Auto-filling missing.")
                    # 补全缺失的章节
                    for i in range(len(batch_chapters), expected):
                        missing_idx = start_num + i
                        batch_chapters.append({
                            "chapter_num": missing_idx,
                            "title": f"第{missing_idx}章",
                            "must_events": [],
                            "forbidden_events": []
                        })
                elif len(batch_chapters) > expected:
                    # 截断多余的
                    batch_chapters = batch_chapters[:expected]
                
                # 确保 chapter_num 正确
                for i, ch in enumerate(batch_chapters):
                    expected_num = start_num + i
                    if ch.get("chapter_num") != expected_num:
                        logger.warning(f"Correcting chapter_num from {ch.get('chapter_num')} to {expected_num}")
                        ch["chapter_num"] = expected_num
                
                all_chapters.extend(batch_chapters)
                # 避免请求过快
                await asyncio.sleep(1)
            
            vol["chapters"] = all_chapters
            full_outline["volumes"].append(vol)
            logger.info(f"Generated {len(all_chapters)} chapters for volume {vol['volume_num']}")

        logger.info(f"Stepwise outline generated: {len(full_outline['volumes'])} volumes, total chapters={sum(len(v['chapters']) for v in full_outline['volumes'])}")
        return full_outline

    # ========== 以下是原有的辅助方法，保持不变 ==========
    async def plan_request_with_prompt(self, prompt: str, task_type: str) -> str:
        """直接发送自定义 prompt 给 LLM，返回原始响应。"""
        router = get_router()
        pool = get_llm_router_pool()

        # 根据任务类型选择模型和 max_tokens
        if task_type in ("scene_plan", "novel_outline", "volumes_outline"):
            model_name = router.get_model_for_task("plan")
            candidates = [model_name]
            max_tokens = 8192
        elif task_type == "chapters_outline":
            model_name = router.get_model_for_task("plan")
            candidates = [model_name]
            max_tokens = 24576   # 章节生成需要更大输出
        elif task_type == "code":
            model_name = router.get_model_for_task("code")
            candidates = [model_name]
            max_tokens = 8192
        else:
            model_name = router.get_model_for_task("default")
            candidates = [model_name]
            max_tokens = 8192

        async def _call_llm(model, *args, **kwargs):
            logger.info(f"Planner using model: {model}")
            api_url = kwargs.get('base_url', config.llm_api_url)
            # 增加超时时间到 1 小时
            timeout = httpx.Timeout(3600.0, connect=60.0)
            client = AsyncOpenAI(api_key="not-needed", base_url=api_url, timeout=timeout)
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=max_tokens
                )
                message = response.choices[0].message
                content = message.content or ""
                if not content and hasattr(message, "reasoning_content"):
                    content = message.reasoning_content or ""
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
        return await pool.call_with_fallback(candidates, _call_llm, timeout=config.llm_timeout_planning, agent="planner")

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
    
    def _build_narrative_intent_from_scene(
        self,
        scene: Dict[str, Any],
        scene_id: str,
        chapter: int,
        scene_idx: int,
    ) -> NarrativeIntent:
        """
        从场景数据构建 NarrativeIntent。
        当前为启发式构建，后续由 Planner LLM 直接输出。
        """
        goal = scene.get("goal", "推进剧情")
        scene_spec = scene.get("scene_spec", {})
        narrative_function = scene_spec.get("narrative_function", "")

        # 推断 SceneRole
        role_map = {
            "introduce_mystery": SceneRole.SETUP,
            "escalate": SceneRole.CONFLICT_ESCALATION,
            "reveal_truth": SceneRole.DISCOVERY,
            "release_tension": SceneRole.RECOVERY,
            "transition": SceneRole.TRANSITION,
            "foreshadow": SceneRole.CLIMAX_PREPARATION,
        }
        scene_role = role_map.get(narrative_function, SceneRole.TRANSITION)

        # 生成确定性 intent_id
        intent_id = NarrativeIntent.generate_intent_id(
            scene_id=scene_id,
            role=scene_role,
            objective=goal
        )

        return NarrativeIntent(
            intent_id=intent_id,
            scene_role=scene_role,
            objective=goal,
            preconditions=[],      # 暂由 Planner LLM 填充
            beats=[],              # 暂由 Planner LLM 填充
            consequences=[],       # 暂由 Planner LLM 填充
            interaction_plan=None,
        )