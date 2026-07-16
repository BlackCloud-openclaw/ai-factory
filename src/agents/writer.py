"""
Writer Agent - 基于状态驱动的章节生成（强制服从 state_delta）
集成 Phase 6 Runtime Pipeline（简化版）
"""
import json
import time
import re
import pathlib
from typing import Dict, Any, Optional
from openai import AsyncOpenAI

from src.agents.base import BaseAgent
from src.orchestrator.state import AgentState
from src.execution.llm_router_pool import get_llm_router_pool
from src.model_router import get_router
from src.config import config
from src.common.logging import setup_logging
from src.common.retry import retry_with_backoff

from src.writing.voiceprint import VoiceprintRegistry
from src.writing.context_compiler import ContextCompiler
from src.writing.prompt_firewall import PromptFirewall
from src.writing.validators import validate_all
from src.writing.world_state import WorldState
from src.common.prompt_logger import log_prompt
from src.writing.history_summarizer import get_chapter_summaries, get_key_events_summary
from src.writing.voice_memory import VoiceMemory
from src.writing.narrative_projection import NarrativeProjector
from src.writing.planning_contract import PlanningContract

logger = setup_logging("agents.writer")

# ========== Phase 6: Runtime 模块（简化版） ==========
try:
    from src.runtime import validate_draft
    RUNTIME_AVAILABLE = True
    logger.info("Runtime module available")
except ImportError as e:
    RUNTIME_AVAILABLE = False
    logger.warning(f"Runtime module not available: {e}")

# 全局声纹注册表
_voiceprint_registry = None

def get_voiceprint_registry():
    global _voiceprint_registry
    if _voiceprint_registry is None:
        _voiceprint_registry = VoiceprintRegistry("config/voiceprints.yaml")
    return _voiceprint_registry


class WritingAgent(BaseAgent):
    task_type = "writing"

    # Grammar 缓存
    _grammar: Optional[str] = None

    @classmethod
    def _get_grammar(cls) -> Optional[str]:
        """加载 GBNF grammar 文件，用于强制 JSON 输出"""
        if cls._grammar is not None:
            return cls._grammar if cls._grammar else None

        grammar_path = pathlib.Path(__file__).parent.parent.parent / "grammars" / "json_writer.gbnf"
        if grammar_path.exists():
            try:
                cls._grammar = grammar_path.read_text(encoding="utf-8")
                logger.info(f"Loaded grammar from {grammar_path}")
                return cls._grammar
            except Exception as e:
                logger.error(f"Failed to load grammar: {e}")
                cls._grammar = ""  # 标记为已尝试但失败
        else:
            logger.warning(f"Grammar file not found at {grammar_path}, JSON validation will rely on model only.")
            cls._grammar = ""
        return None

    def _fix_truncated_json(self, text: str) -> str:
        if not text:
            return text
        # 移除 markdown 代码块标记
        text = re.sub(r'^```json\s*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
        # 补全缺失的括号
        brace_diff = text.count('{') - text.count('}')
        bracket_diff = text.count('[') - text.count(']')
        if bracket_diff > 0:
            text += ']' * bracket_diff
        if brace_diff > 0:
            text += '}' * brace_diff
        if text.endswith(','):
            text = text[:-1]
            if text.endswith('{'):
                text += '}'
            elif text.endswith('['):
                text += ']'
        return text

    async def run(self, state: AgentState) -> Dict[str, Any]:
        self._state = state
        logger.info("WritingAgent starting")
        start_time = time.time()

        scene_plan = state.scene_plan or {}
        constraints = state.writing_constraints or {}

        # ========== 提取 Planning Contract ==========
        planning_contract = None
        if hasattr(state, 'planning_contract') and state.planning_contract:
            try:
                contract_data = state.planning_contract
                planning_contract = PlanningContract(**contract_data)
                logger.info(f"✅ Loaded Planning Contract for writer: {planning_contract.scene_id}")
            except Exception as e:
                logger.warning(f"Failed to parse Planning Contract: {e}, falling back to scene_plan")
                planning_contract = None

        # ========== 读取 Director 输出 ==========
        narrative_blueprint = state.narrative_blueprint or {}
        knowledge_deltas = state.knowledge_deltas or []
        character_intent = state.character_intent or {}

        # ========== 1. 提取强制 state_delta ==========
        forced_delta = scene_plan.get("state_delta", {})
        expected_events = forced_delta.get("events", [])

        voiceprint_registry = get_voiceprint_registry()
        world_state = WorldState.from_dict(state.current_state) if state.current_state else WorldState()

        compiler = ContextCompiler(max_tokens=2000)

        # 实验模式：检查 metadata 中是否有 planning_directive
        if state.metadata and "planning_directive" in state.metadata:
            from src.writing.planning_directive import PlanningDirective
            directive_data = state.metadata["planning_directive"]
            directive = PlanningDirective(**directive_data)
            compiler.set_planning_directive(directive)
            logger.info(f"实验模式：注入 {directive.representation} 规划指令")

        compiled_context = compiler.compile(
            world_state,
            active_characters=scene_plan.get("characters", []),
            max_active=10
        )

        history_summaries = []
        key_events_summary = None
        if state.novel_id:
            history_summaries = await get_chapter_summaries(
                state.novel_id, state.current_volume, state.current_chapter
            )
            key_events_summary = await get_key_events_summary(
                state.novel_id, state.current_volume, state.current_chapter
            )

        # ========== 恢复 voice_memory ==========
        voice_memory = None
        if state.voice_memory:
            voice_memory = VoiceMemory(state.novel_id, state.voice_memory)
        elif state.compressed_state and "voice_fingerprint" in state.compressed_state:
            voice_memory = VoiceMemory(state.novel_id, state.compressed_state["voice_fingerprint"])
        else:
            voice_memory = VoiceMemory(state.novel_id)

        prompt = compiler.build_writer_prompt(
            scene_plan=scene_plan,
            world_state=world_state,
            voiceprint_registry=voiceprint_registry,
            compiled_context=compiled_context,
            history_summaries=history_summaries,
            key_events_summary=key_events_summary,
            voice_memory=voice_memory,
        )

        # ========== 新增：如果存在 Contract，用 Contract 信息增强 Prompt ==========
        if planning_contract:
            contract_prompt = self._build_contract_prompt(planning_contract)
            prompt += "\n\n" + contract_prompt

        # ====== 注入戏剧结构（来自 Drama Planner） ======
        drama_structure = state.drama_structure or {}
        if drama_structure:
            drama_text = self._format_drama_structure(drama_structure)
            prompt += f"\n\n【🎭 戏剧结构强制约束 - 必须严格遵循】\n{drama_text}"
            prompt += "\n\n【🔊 对话要求（强制执行）】\n"
            prompt += "1. 本场景必须包含至少 **5 轮有效对话**（每轮对话指一方发言及对方的回应）。\n"
            prompt += "2. 对话总字数应占场景正文总字数的 **30%~50%**，确保充分展现角色间的冲突、压力和决策。\n"
            prompt += "3. 对话内容应围绕戏剧结构中的**欲望、阻碍、压力、选择、代价**展开，通过语言交锋体现角色的博弈。\n"
            prompt += "4. 避免纯叙述性描写，优先使用对话来推动情节和揭示信息。\n"
            prompt += "\n\n【🔊 对话模板示例（必须参考）】\n"
            prompt += "对话不应是寒暄，而应是交锋。例如：\n"
            prompt += "- 角色 A：「你以为这样就能拦得住我？」\n"
            prompt += "- 角色 B：「试试看。」（拔剑）\n"
            prompt += "- 角色 A：「有意思...那我就陪你玩玩。」\n"

        # ========== 2. 添加强制 state_delta 指令 ==========
        if expected_events:
            prompt += "\n\n【🔒 强制状态变更（必须原样输出到 events 字段，不得增删改）】\n"
            prompt += json.dumps(forced_delta, ensure_ascii=False, indent=2)
            prompt += "\n你必须将以上 JSON 对象中的 'events' 数组原封不动地放入输出 JSON 的 'events' 字段中。\n"
            prompt += "不允许添加、删除或修改任何事件。"

        # ========== 3. 注入验证反馈 ==========
        feedback = state.metadata.get("writing_feedback", "")
        if feedback:
            if "知识变化未体现" in feedback:
                missing_infos = re.findall(r'知识变化未体现: ([^(]+)', feedback)
                if missing_infos:
                    prompt += "\n\n【🔴 重试强制要求】上一次验证失败，缺失以下知识变化：\n"
                    for info in missing_infos:
                        prompt += f"- {info.strip()}\n"
                    prompt += "你必须在下一次生成的 scene_text 中，**以独立句子明确写出**上述信息，不可省略或暗示。\n"
                else:
                    prompt += f"\n\n【⚠️ 上一次生成失败，请根据以下反馈修正】\n{feedback}\n"
                    prompt += "请仔细阅读反馈，修正正文中的问题，并确保强制事件原样输出。\n"
            else:
                prompt += f"\n\n【⚠️ 上一次生成失败，请根据以下反馈修正】\n{feedback}\n"
                prompt += "请仔细阅读反馈，修正正文中的问题，并确保强制事件原样输出。\n"

        if constraints.get("forbidden_events"):
            prompt += f"\n\n【禁止事件】\n" + "\n".join(constraints["forbidden_events"])

        # ========== 4. 注入 Director 蓝图指令 ==========
        director_instructions = ""
        if narrative_blueprint:
            director_instructions = f"""
    \n\n【🎬 导演蓝图 - 必须严格遵循】
    - 注意力轨迹: {narrative_blueprint.get('attention_path', [])}
    - 延迟的信息: {narrative_blueprint.get('withheld_information', '')}
    - 揭示节拍: {narrative_blueprint.get('reveal_beat', '')}
    - 压力来源: {narrative_blueprint.get('scene_pressure', '')}
    - 沉默动作优先级: {narrative_blueprint.get('silent_action_priority', '')}
    - 重复意象: {narrative_blueprint.get('recurring_image', '')}
    - 场景角色: {narrative_blueprint.get('scene_role', 'SETUP')}
    """

        if knowledge_deltas:
            director_instructions += """
    【📖 知识变化软约束 - 必须覆盖核心语义】
    以下是导演要求在本场景中向读者释放的信息。你必须在 scene_text 中**通过自然的方式明确体现每条信息的核心语义**（可通过直接陈述、角色对话、内心独白、环境描写等），不得遗漏。你可以自由组织语言，但必须让读者能够清晰接收到这些信息。

    """
            for i, kd in enumerate(knowledge_deltas, 1):
                info = kd.get('information', '')
                if info:
                    director_instructions += f"{i}. {info}\n"
            director_instructions += """
    示例：对于“灵气逆向运转会导致经脉重塑”，你可以写“他察觉到灵气在体内逆向流动，经脉隐隐有被重塑的感觉”。
    注意：不要只通过隐晦暗示，要让信息明确可辨。
    """

        if character_intent:
            director_instructions += f"""
    【角色意图 - 不可违背】
    {json.dumps(character_intent, ensure_ascii=False, indent=2)}
    """

        prompt += director_instructions

        # ========== 实验注入：叙事投影 (Loop/Focus) ==========
        if state.novel_id:
            latest = await NarrativeProjector.get_latest(state.novel_id)
            if latest:
                projection = latest.to_dict()
                experiment_group = state.metadata.get("experiment_group", "baseline")
                logger.info(f"📊 实验组: {experiment_group}")

                if experiment_group == "loop":
                    loop = projection.get("loop", {})
                    if loop:
                        prompt += f"""
    【叙事循环 (Narrative Loop) - 实验注入】
    - 当前正在推进的过程：{loop.get('description', '')}
    - 推动者：{loop.get('initiator', '')}
    - 紧迫度：{loop.get('urgency', 0.5)}
    请确保当前场景的推进方向与上述循环保持一致。
    """

                elif experiment_group == "focus":
                    focus = projection.get("focus", {})
                    if focus:
                        prompt += f"""
    【叙事聚焦 (Narrative Focus) - 实验注入】
    - 当前最重要的未完成事项：{focus.get('subject', '')}
    - 类型：{focus.get('type', '')}
    - 为什么重要：{focus.get('why_matters', '')}
    请确保当前场景的行动与上述聚焦事项相关。
    """

                elif experiment_group == "both":
                    focus = projection.get("focus", {})
                    loop = projection.get("loop", {})
                    if focus or loop:
                        prompt += f"""
    【叙事循环 + 聚焦 - 实验注入】
    循环：{loop.get('description', '')}（紧迫度：{loop.get('urgency', 0.5)}）
    聚焦：{focus.get('subject', '')}（类型：{focus.get('type', '')}）
    请确保当前场景的推进方向与上述循环和聚焦保持一致。
    """

                elif experiment_group == "full":
                    projection_data = projection
                    prompt += f"""
    【完整叙事状态 - 实验注入】
    - 循环：{projection_data.get('loop', {}).get('description', '')}
    - 聚焦：{projection_data.get('focus', {}).get('subject', '')}
    - 注意力：{projection_data.get('attention', {}).get('target', '')}
    - 核心问题：{projection_data.get('question', {}).get('text', '')}
    请基于以上叙事状态驱动当前场景的写作。
    """

        # ========== 调用 LLM ==========
        router = get_router()
        primary_model = router.get_model_for_task("writing")
        fallback_model = "Qwen3-32B-Q5_K_M"
        pool = get_llm_router_pool()
        raw_output = None
        last_error = None

        try:
            raw_output = await pool.call(
                primary_model,
                self._call_llm,
                prompt,
                timeout=getattr(config, 'llm_timeout_writing', 600),
                agent="writer"
            )
        except Exception as e:
            if "truncated" in str(e) or "missing closing brace" in str(e):
                logger.warning(f"Output truncated for primary model {primary_model}, retrying with larger context...")
            logger.warning(f"Primary model {primary_model} failed: {e}, trying fallback")
            last_error = e
            try:
                raw_output = await pool.call(
                    fallback_model,
                    self._call_llm,
                    prompt,
                    timeout=getattr(config, 'llm_timeout_writing', 600),
                    agent="writer"
                )
                logger.info(f"Fallback model {fallback_model} succeeded")
            except Exception as e2:
                logger.error(f"Fallback model also failed: {e2}")
                raw_output = f'{{"scene_text": "[生成失败: {last_error}]", "events": [], "foreshadowing": []}}'

        logger.info(f"Raw LLM output: {raw_output[:500]}")

        fixed = self._fix_truncated_json(raw_output)
        try:
            data = json.loads(fixed)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}, attempting regex fallback")
            match = re.search(r'"scene_text"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_output, re.DOTALL)
            if match:
                scene_text = match.group(1).replace('\\"', '"').replace('\\n', '\n')
                data = {"scene_text": scene_text, "events": [], "foreshadowing": []}
                logger.info(f"Extracted scene_text via regex (length={len(scene_text)})")
            else:
                data = {"scene_text": "[生成失败: JSON解析错误]", "events": [], "foreshadowing": []}
                logger.error("Failed to extract scene_text even with regex")

        # ========== 验证输出 ==========
        must_events = scene_plan.get("must_events", [])
        context = {"must_events": must_events}
        validation = validate_all(raw_output, context, async_semantic=False)

        scene_text = ""
        parsed = validation.get("parsed_output")
        deviation_detected = False
        error_msg = None

        if validation["passed"]:
            parsed = validation["parsed_output"] or {}
            scene_text = parsed.get("scene_text", "")

            events = parsed.get("events", [])
            if events:
                for evt in events:
                    if evt.get("type") == "discovery" and "importance" in evt:
                        imp = evt["importance"]
                        if isinstance(imp, int):
                            if imp >= 5:
                                evt["importance"] = "critical"
                            elif imp >= 3:
                                evt["importance"] = "high"
                            elif imp >= 1:
                                evt["importance"] = "normal"
                            else:
                                evt["importance"] = "low"
                        elif isinstance(imp, float):
                            evt["importance"] = "critical" if imp >= 5 else "high" if imp >= 3 else "normal" if imp >= 1 else "low"
                        elif isinstance(imp, bool):
                            evt["importance"] = "critical" if imp else "low"
                    if evt.get("type") == "relationship_change":
                        if "new_value" not in evt and "delta" in evt:
                            from_char = evt.get("from_char")
                            to_char = evt.get("to_char")
                            if from_char and to_char:
                                key = f"{from_char}|{to_char}"
                                old_value = world_state.relationships.get(key, 0) if world_state else 0
                                delta = evt["delta"]
                                new_value = old_value + delta
                                new_value = max(-100, min(100, new_value))
                                evt["new_value"] = new_value
                            else:
                                evt["new_value"] = 0
                                logger.warning(f"Missing from_char/to_char in relationship_change event: {evt}")
                    if evt.get("type") == "hp_changed":
                        new_hp = evt.get("new_hp")
                        if isinstance(new_hp, (int, float)) and new_hp < 0:
                            evt["new_hp"] = 0
                            logger.warning(f"Corrected negative hp to 0: {new_hp}")
                    if evt.get("type") == "mp_changed":
                        new_mp = evt.get("new_mp")
                        if isinstance(new_mp, (int, float)) and new_mp < 0:
                            evt["new_mp"] = 0
                            logger.warning(f"Corrected negative mp to 0: {new_mp}")
                parsed["events"] = events

            if expected_events:
                actual_events = parsed.get("events", [])
                if json.dumps(actual_events, sort_keys=True) != json.dumps(expected_events, sort_keys=True):
                    deviation_detected = True
                    error_msg = f"state_delta mismatch: expected {len(expected_events)} events, got {len(actual_events)}"
                    logger.error(error_msg)
        else:
            logger.warning(f"Validation failed: {validation['error']}")
            try:
                match = re.search(r'"scene_text"\s*:\s*"([^"]*)"', raw_output)
                scene_text = match.group(1) if match else f"[验证失败: {validation['error']}]"
            except:
                scene_text = f"[验证失败: {validation['error']}]"

        missing_goal = []
        missing_conflict = []

        duration = time.time() - start_time
        logger.info(f"WritingAgent completed, duration={duration:.2f}")

        # ========== 更新风格指纹 ==========
        if scene_text:
            voice_memory.update_from_chapter(scene_text)

        # ========== 后处理清洗 ==========
        scene_text = re.sub(r'推进主线剧情\(场景\d+\)[！!]?', '', scene_text)
        scene_text = re.sub(r"'([^']*)'", r"“\1”", scene_text)
        scene_text = re.sub(r'\n{3,}', '\n\n', scene_text)

        # 然后将清洗后的 scene_text 放回 parsed
        if parsed:
            parsed["scene_text"] = scene_text

        # ============================================================
        # ========== Phase 6.5: 消费 ExecutionResult ==========
        # ============================================================
# src/agents/writer.py（仅 Phase 6.5 消费部分）

        # ============================================================
        # ========== Phase 6.5: 消费 ExecutionResult ==========
        # ============================================================
        if RUNTIME_AVAILABLE and scene_text and len(scene_text) > 50:
            try:
                from src.workflow.revision_workflow import RevisionWorkflow

                # 定义 LLM 适配器
                async def llm_adapter(prompt: str) -> str:
                    try:
                        return await pool.call(primary_model, self._call_llm, prompt, agent="writer")
                    except Exception as e:
                        logger.warning(f"Primary model failed in revision: {e}, trying fallback")
                        return await pool.call(fallback_model, self._call_llm, prompt, agent="writer")

                workflow = RevisionWorkflow(
                    llm_executor=llm_adapter,
                    layer_targets=None,
                    max_rounds=2,
                    compliance_threshold=0.7,
                    enable_revision=True,
                )

                result = await workflow.execute(scene_text)

                # ---- Phase 6.5/6.5.1: 打印 ExecutionResult 摘要 + Layer 详情 ----
                stages_summary = []
                for stage in result.get("stages", []):
                    status = stage.get("status", "unknown")
                    payload = stage.get("payload", {})
                    if stage["stage"] == "validation":
                        stages_summary.append(f"validation={payload.get('compliance', 0):.2f}")
                        # ---- Phase 6.5.1: 打印 Layer 详情 ----
                        layers = payload.get("layers", [])
                        for layer_info in layers:
                            missing = layer_info.get("missing", [])
                            observed = layer_info.get("observed", [])
                            if missing:
                                logger.info(f"  {layer_info['layer']}: missing {', '.join(missing)} (observed: {', '.join(observed) if observed else 'none'})")
                            else:
                                logger.info(f"  {layer_info['layer']}: ✅ (observed: {', '.join(observed) if observed else 'none'})")
                    elif stage["stage"] == "edit_plan":
                        stages_summary.append(f"edit_plan={payload.get('action_count', 0)}")
                    elif stage["stage"] == "patch_render":
                        stages_summary.append(f"patch_render={payload.get('prompt_length', 0)}")
                    elif stage["stage"] == "llm":
                        stages_summary.append(f"llm={status}, finish_reason={payload.get('finish_reason', 'unknown')}")
                    elif stage["stage"] == "revalidation":
                        stages_summary.append(f"revalidation={payload.get('final_compliance', 0):.2f}")

                delta = result.get("compliance_delta", 0.0)
                logger.info(f"ExecutionResult: {', '.join(stages_summary)} | delta={delta:+.2f}")

                # 更新场景文本
                scene_text = result.get("final_text", scene_text)
                compliance = result.get("compliance", 1.0)

                # 保存到数据库
                if hasattr(state, 'novel_id') and state.novel_id:
                    await self._save_runtime_report(
                        novel_id=state.novel_id,
                        volume_num=state.current_volume or 1,
                        chapter_num=state.current_chapter or 1,
                        scene_idx=state.current_scene_index or 0,
                        scene_text=scene_text,
                        validation_result={
                            "compliance": compliance,
                            "delta": result.get("compliance_delta", 0.0),
                            "stages": result.get("stages", []),
                        }
                    )

                if compliance < 0.7:
                    logger.warning(f"Low compliance ({compliance:.2f}) for {state.novel_id}")
                else:
                    logger.info(f"Runtime compliance: {compliance:.2f}")

            except Exception as e:
                logger.warning(f"Runtime workflow failed: {e}")

        return {
            "scene_text": raw_output,
            "final_answer": scene_text,
            "deviation_detected": deviation_detected,
            "missing_goal_keywords": missing_goal,
            "missing_conflict_keywords": missing_conflict,
            "error": error_msg,
            "narrative_blueprint": narrative_blueprint,
            "knowledge_deltas": knowledge_deltas,
            "character_intent": character_intent,
            "voice_memory": voice_memory.to_dict(),
        }

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    async def _call_llm(self, model_name: str, prompt: str, **kwargs) -> str:
        metadata = {
            "model": model_name,
            "temperature": 0.7,
            "max_tokens": 32768,
        }
        if hasattr(self, '_state') and self._state:
            metadata.update({
                "novel_id": getattr(self._state, 'novel_id', None),
                "chapter": getattr(self._state, 'current_chapter', None),
                "scene": getattr(self._state, 'current_scene_index', None),
            })

        constraints = None
        if hasattr(self, '_state') and self._state and hasattr(self._state, 'current_state'):
            from src.writing.state_projection import extract_hard_constraints
            from src.writing.world_state import WorldState
            world = WorldState.from_dict(self._state.current_state) if self._state.current_state else None
            if world:
                constraints = extract_hard_constraints(world)

        log_prompt("writer", prompt, metadata, constraints=constraints)

        actual_model = model_name
        base_url = kwargs.get('base_url')
        if not base_url:
            pool = get_llm_router_pool()
            base_url = pool.get_base_url(actual_model)

        client = AsyncOpenAI(api_key="not-needed", base_url=base_url)

        extra_body = {}
        grammar_str = self._get_grammar()
        if grammar_str:
            extra_body["grammar"] = grammar_str

        max_tokens = 32768
        temperature = 0.7

        # ---- LLM Request 观测 ----
        logger.info(
            "[LLM] request model=%s temperature=%.2f max_tokens=%d prompt_chars=%d prompt_lines=%d grammar_sent=%s extra_body_keys=%s",
            actual_model,
            temperature,
            max_tokens,
            len(prompt),
            prompt.count("\n") + 1,
            bool(grammar_str),
            list(extra_body.keys()) if extra_body else [],
        )

        response = await client.chat.completions.create(
            model=actual_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body if extra_body else None,
        )

        choice = response.choices[0]
        finish_reason = choice.finish_reason or "unknown"
        content = choice.message.content or ""

        usage = getattr(response, "usage", None)

        # ---- LLM Response 观测 ----
        if usage:
            logger.info(
                "[LLM] finish_reason=%s output_chars=%d usage(prompt=%s completion=%s total=%s)",
                finish_reason,
                len(content),
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )
        else:
            logger.info(
                "[LLM] finish_reason=%s output_chars=%d usage=unavailable",
                finish_reason,
                len(content),
            )

        logger.debug("[LLM] tail=%r", content[-200:])

        # ---- JSON 完整性检查与修复 ----
        stripped = content.strip()
        json_repair_attempted = False
        json_repair_changed = False
        json_parse_success = False

        if not (stripped.endswith('}') or stripped.endswith(']')):
            json_repair_attempted = True
            fixed = self._fix_truncated_json(content)
            json_repair_changed = (fixed != content)
            logger.info(
                "[LLM] json_repair attempted=%s changed=%s",
                json_repair_attempted,
                json_repair_changed,
            )
            if json_repair_changed:
                logger.info("[LLM] json repair changed length %d -> %d", len(content), len(fixed))
                content = fixed
            else:
                logger.error("[LLM] json repair failed, original tail: %r", content[-200:])

        # 验证是否可解析
        try:
            import json
            json.loads(content)
            json_parse_success = True
        except json.JSONDecodeError:
            json_parse_success = False

        logger.info(
            "[LLM] parse_success=%s final_length=%d",
            json_parse_success,
            len(content),
        )

        # ---- 新增：保存 Prompt + Response（按分类） ----
        import os
        from datetime import datetime

        def _save_prompt_response(prompt_text: str, response_text: str, parse_success: bool, model: str):
            logs_dir = "logs/llm_artifacts"
            os.makedirs(logs_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            status = "success" if parse_success else "failed"
            trace_id = f"{timestamp}_{status}_{model.replace('/', '_')}"
            
            prompt_path = os.path.join(logs_dir, f"{trace_id}_prompt.txt")
            response_path = os.path.join(logs_dir, f"{trace_id}_response.txt")
            
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt_text)
            with open(response_path, "w", encoding="utf-8") as f:
                f.write(response_text)
            
            logger.info(f"[LLM] artifacts saved: {trace_id}_prompt.txt / {trace_id}_response.txt")

        _save_prompt_response(prompt, content, json_parse_success, actual_model)

        # 如果仍然无法解析，抛出异常
        if not json_parse_success:
            raise ValueError("Response invalid JSON: missing closing brace or malformed")

        logger.info(f"Received response from {actual_model}, length={len(content)}")
        return content 

    # ============================================================
    # 辅助方法
    # ============================================================

    def _format_drama_structure(self, ds: Dict[str, Any]) -> str:
        """将 Drama Planner 的 JSON 转为 LLM 可读的指令"""
        parts = []
        if ds.get("scene_goal"):
            parts.append(f"🟢 核心欲望：{ds['scene_goal']}")
        obstacle = ds.get("obstacle", {})
        if obstacle.get("description"):
            parts.append(f"🔴 阻碍（必须克服）：{obstacle['description']}（类型：{obstacle.get('type', '未知')}）")
        pressure = ds.get("pressure", {})
        if pressure.get("description"):
            parts.append(f"⏳ 压力（紧迫性）：{pressure['description']}（类型：{pressure.get('type', '未知')}）")
        decision = ds.get("decision", {})
        if decision.get("options"):
            parts.append(f"⚖️ 两难选择（必须二选一）：{', '.join(decision['options'])}")
        if decision.get("chosen"):
            parts.append(f"👉 最终决定：{decision['chosen']}")
        cost = ds.get("cost", {})
        cost_parts = []
        if cost.get("success"):
            cost_parts.append(f"成功代价：{cost['success']}")
        if cost.get("failure"):
            cost_parts.append(f"失败代价：{cost['failure']}")
        if cost_parts:
            parts.append(f"💔 代价：{'；'.join(cost_parts)}")
        rel_delta = ds.get("relationship_delta", {})
        if rel_delta.get("target") and rel_delta.get("to"):
            parts.append(f"🔄 关系变化：与 {rel_delta['target']} 的关系从 {rel_delta.get('from', '当前状态')} 变为 {rel_delta['to']}")
        if not parts:
            return "(空)"
        return "\n".join(parts)

    def _build_contract_prompt(self, contract: PlanningContract) -> str:
        """根据 Planning Contract 生成写作指令"""
        lines = []
        lines.append("【📋 规划契约（Planning Contract）】")
        lines.append(f"场景目标：{contract.intent.goal}")
        lines.append(f"核心冲突：{contract.intent.conflict}")
        lines.append(f"预期结果：{contract.intent.expected_outcome}")

        if contract.execution.units:
            lines.append("\n必须完成的执行单元：")
            for unit in contract.execution.units:
                lines.append(f"- {unit.label}: {unit.description}")

        if contract.constraints:
            lines.append("\n硬性约束：")
            for c in contract.constraints:
                if c.type == "required":
                    lines.append(f"  ✅ 必须发生：{c.target}")
                elif c.type == "forbidden":
                    lines.append(f"  ❌ 禁止发生：{c.target}")
                elif c.type == "before":
                    lines.append(f"  ⏳ {c.target} 必须在 {c.condition} 之前")
                elif c.type == "after":
                    lines.append(f"  ⏳ {c.target} 必须在 {c.condition} 之后")
                elif c.type == "exclusive":
                    lines.append(f"  🔀 {c.target} 互斥")
                elif c.type == "at_least_once":
                    lines.append(f"  🔁 {c.target} 至少发生一次")

        if contract.observables.state_changes:
            lines.append("\n场景结束后世界状态应发生变化：")
            for change in contract.observables.state_changes:
                if change.type == "plot_flag":
                    lines.append(f"- 剧情标记 {change.name} 应为 {change.value}")
                elif change.type == "relationship":
                    lines.append(f"- {change.from_char} 与 {change.to_char} 的关系变化 {change.delta}")
                elif change.type == "inventory":
                    lines.append(f"- {change.actor} {change.operation} {change.item}")
                elif change.type == "realm":
                    lines.append(f"- {change.actor} 突破至 {change.to_major_realm}{change.to_minor_stage}层")
                elif change.type == "location":
                    lines.append(f"- {change.actor} 进入 {change.location}")
                elif change.type == "hp":
                    lines.append(f"- {change.actor} HP 变为 {change.new_hp}")

        lines.append("\n⚠️ 请严格遵循以上契约，尤其是执行单元和硬性约束。")
        return "\n".join(lines)

    # ============================================================
    # Phase 6 Runtime 辅助方法
    # ============================================================

    async def _save_runtime_report(
        self,
        novel_id: str,
        volume_num: int,
        chapter_num: int,
        scene_idx: int,
        scene_text: str,
        validation_result: dict
    ) -> None:
        """将 Runtime 分析结果保存到 narrative_versions 表"""
        from src.db.pool import get_db_pool
        import json

        pool = get_db_pool()
        if not pool:
            logger.warning("Database pool not available, skipping Runtime report save")
            return

        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO narrative_versions
                (novel_id, volume_num, chapter_num, scene_idx, version_type,
                 scene_text, kpi_scores, generated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                ON CONFLICT (novel_id, volume_num, chapter_num, scene_idx, version_type)
                DO UPDATE SET
                    scene_text = EXCLUDED.scene_text,
                    kpi_scores = EXCLUDED.kpi_scores,
                    generated_at = NOW()
            """, novel_id, volume_num, chapter_num, scene_idx, "A",
                scene_text, json.dumps(validation_result))

            logger.debug(f"Saved Runtime report to narrative_versions for {novel_id}")