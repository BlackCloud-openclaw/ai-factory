"""
Writer Agent - 基于状态驱动的章节生成（强制服从 state_delta）
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

logger = setup_logging("agents.writer")

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

    def _fix_truncated_json(self, text: str, aggressive: bool = False) -> str:
        """修复截断的 JSON，自动补全缺失的括号，并尝试移除最后一个不完整的字段"""
        if not text:
            return text
        
        # 统计括号
        brace_count = text.count('{') - text.count('}')
        bracket_count = text.count('[') - text.count(']')
        
        # 如果括号平衡，直接返回
        if brace_count <= 0 and bracket_count <= 0 and not aggressive:
            return text
        
        # 尝试移除最后一个不完整的字段（aggressive模式）
        if aggressive and brace_count > 0:
            # 找到最后一个逗号的位置，删除之后的内容
            last_comma = text.rfind(',')
            if last_comma > 0:
                # 检查最后一个逗号后面是否有完整的字段
                after_comma = text[last_comma+1:].strip()
                # 如果后面没有闭合括号，说明字段不完整
                if after_comma and not (after_comma.endswith('}') or after_comma.endswith(']')):
                    text = text[:last_comma]
                    # 重新计算括号
                    brace_count = text.count('{') - text.count('}')
                    bracket_count = text.count('[') - text.count(']')
        
        # 补全缺失的 }
        if brace_count > 0:
            text += '}' * brace_count
        
        # 补全缺失的 ]
        if bracket_count > 0:
            text += ']' * bracket_count
        
        # 额外检查：如果最后一个字符是逗号，移除它
        if text.endswith(','):
            text = text[:-1]
            # 如果移除后末尾是 { 或 [，补全
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

        # ========== 恢复 voice_memory（优先从 compressed_state 加载）==========
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
                import re
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
                    agent="writer"           # 建议补上
                )
                logger.info(f"Fallback model {fallback_model} succeeded")
            except Exception as e2:
                logger.error(f"Fallback model also failed: {e2}")
                raw_output = f'{{"scene_text": "[生成失败: {last_error}]", "events": [], "foreshadowing": []}}'

        logger.info(f"Raw LLM output: {raw_output[:500]}")

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

            # 事件格式修复
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

        # ========== 更新风格指纹（每个场景） ==========
        if scene_text:
            voice_memory.update_from_chapter(scene_text)

        # ========== 新增：后处理清洗 ==========
        # 1. 删除 "推进主线剧情（场景X）" 占位符（包括感叹号）
        scene_text = re.sub(r'推进主线剧情（场景\d+）[！!]?', '', scene_text)
        # 2. 统一引号为中文双引号（将单引号包裹的对话转为双引号）
        # 简单规则：将 '...' 形式的对话转为 “...” 
        # 注意：不处理缩写或所有格，中文场景下几乎都是对话
        scene_text = re.sub(r"'([^']*)'", r"“\1”", scene_text)
        # 3. 可选：去除多余空行
        scene_text = re.sub(r'\n{3,}', '\n\n', scene_text)
        # ====================================

        # 然后将清洗后的 scene_text 放回 parsed（或直接使用）

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
            "max_tokens": 24576,          # 增加到 24576 避免截断
        }
        if hasattr(self, '_state') and self._state:
            metadata.update({
                "novel_id": getattr(self._state, 'novel_id', None),
                "chapter": getattr(self._state, 'current_chapter', None),
                "scene": getattr(self._state, 'current_scene_index', None),
            })
        
        # 提取约束用于哈希
        constraints = None
        if hasattr(self, '_state') and self._state and hasattr(self._state, 'current_state'):
            from src.writing.state_projection import extract_hard_constraints
            from src.writing.world_state import WorldState
            world = WorldState.from_dict(self._state.current_state) if self._state.current_state else None
            if world:
                constraints = extract_hard_constraints(world)
        
        log_prompt("writer", prompt, metadata, constraints=constraints)
        
        # 调用 LLM
        actual_model = model_name
        base_url = kwargs.get('base_url')
        if not base_url:
            pool = get_llm_router_pool()
            base_url = pool.get_base_url(actual_model)
        
        client = AsyncOpenAI(api_key="not-needed", base_url=base_url)
        
        # 添加 grammar 约束
        extra_body = {}
        grammar_str = self._get_grammar()
        if grammar_str:
            extra_body["grammar"] = grammar_str
        
        response = await client.chat.completions.create(
            model=actual_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=24576,               # 与 metadata 保持一致
            extra_body=extra_body if extra_body else None,
        )
        content = response.choices[0].message.content or ""
        
        # ========== JSON 完整性检查与修复 ==========
        # 首先检查是否完整
        stripped = content.strip()
        if not (stripped.endswith('}') or stripped.endswith(']')):
            logger.warning(f"Response from {actual_model} appears truncated (ends with '...{stripped[-50:]}')")
            # 尝试修复
            fixed = self._fix_truncated_json(content, aggressive=True)
            if fixed != content:
                logger.info(f"Successfully fixed truncated JSON (length {len(content)} -> {len(fixed)})")
                content = fixed
            else:
                # 仍然截断，抛出异常触发重试
                raise ValueError("Response truncated, missing closing brace")
        
        logger.info(f"Received response from {actual_model}, length={len(content)}")
        return content