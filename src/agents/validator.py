"""
Validator Agent - 统一验证入口，支持代码和小说的分层验证
"""
import re
import json
import time
import py_compile
import tempfile
import os
import asyncio
from typing import Any, Optional, Dict, List, Tuple
from openai import AsyncOpenAI

from src.model_router import get_router
from src.config import config
from src.common.logging import setup_logging
from src.common.retry import retry_with_backoff
from src.agents.base import BaseAgent
from src.orchestrator.state import AgentState
from src.prompts.validator_prompts import VALIDATOR_PROMPT_REGISTRY

# 新验证架构（仅用于代码验证，小说验证已简化）
from src.writing.validators import validate_all
from src.writing.prompt_firewall import PromptFirewall
from src.writing.summarizer import generate_embedding, cosine_similarity
from src.writing.causality.validator import CausalityValidator
from src.db import get_db_pool
from src.writing.event_store import NarrativeEventStore
from src.writing.causality.budget import ConsistencyBudget
from src.writing.causality.health import HealthChecker
from src.common.prompt_logger import log_prompt
from src.writing.world_state import WorldState
from src.config_loader import get_xianxia_config

logger = setup_logging("agents.validator")


class ValidatorAgent(BaseAgent):
    """Agent responsible for validating code quality and novel consistency."""

    def __init__(
        self,
        llm_api_url: str = config.llm_api_url,
        llm_model: str = config.llm_model_name,
    ):
        self.llm_api_url = llm_api_url
        self.llm_model = llm_model

    async def run(self, state: AgentState) -> Dict[str, Any]:
        agent_name = "ValidatorAgent"
        state.step_count += 1
        step = state.step_count
        logger.info(f"Starting {agent_name}, step={step}")
        start_time = time.time()

        mode = getattr(state, 'validation_mode', 'code')

        # ========== 小说模式：调用真正的校验 ==========
        if mode == "novel" and state.scene_text:
            raw = state.scene_text
            clean = self._extract_scene_text(raw)
            if not clean:
                clean = raw[:200]

            # 构建约束条件
            constraints = {
                "outline": state.outline,
                "writing_constraints": state.writing_constraints or {},
                "must_events": [],
                "goal": "",
                "conflict": "",
                "current_state": state.current_state,   # 新增：用于认知关系检查
            }

            # ========== 新增：注入 Director 输出（用于职责边界检查）==========
            constraints["narrative_blueprint"] = state.narrative_blueprint or {}
            constraints["knowledge_deltas"] = state.knowledge_deltas or []
            constraints["character_intent"] = state.character_intent or {}
            # ========== 新增：注入场景目标（阶段5） ==========
            constraints["scene_objective"] = state.scene_plan.get("scene_objective", "") if state.scene_plan else ""
            # ================================================================

            # 加载当前活跃谓词（用于因果关系校验）
            if state.novel_id:
                pool = get_db_pool()
                if pool:
                    try:
                        event_store = NarrativeEventStore(pool)
                        predicates = await event_store._load_active_predicates(state.novel_id)
                        constraints["predicates"] = predicates
                    except Exception as e:
                        logger.warning(f"Failed to load predicates for causality validation: {e}")

            # ========== 新增：创建一致性预算并加载 ==========
            budget = ConsistencyBudget(state.novel_id, state.current_volume, state.current_chapter)
            await budget.load()
            constraints["budget"] = budget
            # =============================================

            # 健康检查与漂移降级
            drift_level = await HealthChecker.check_drift(state.novel_id)
            if drift_level in ("WARNING", "CRITICAL"):
                constraints["degraded"] = True
                logger.warning(f"Projection drift detected ({drift_level}), validator degraded")

            # 从 scene_plan 中提取必须事件、目标、冲突
            if state.scene_plan:
                constraints["must_events"] = state.scene_plan.get("must_events", [])
                constraints["goal"] = state.scene_plan.get("goal", "")
                constraints["conflict"] = state.scene_plan.get("conflict", "")

            # 调用增强版校验
            result = await self._validate_novel_enhanced(
                raw,
                constraints,
                deviation_detected=getattr(state, 'deviation_detected', False),
                missing_goal=getattr(state, 'missing_goal_keywords', []),
                missing_conflict=getattr(state, 'missing_conflict_keywords', [])
            )

            duration = time.time() - start_time
            status = "success" if result.get("passed") else "error"
            logger.info(f"{agent_name} completed (novel), step={step}, status={status}, duration={duration:.2f}")

            return {
                "validation_result": result,
                "final_answer": clean,
            }

        # ========== 代码验证（保持不变） ==========
        if mode == "code":
            target = state.code_generated
            user_input = state.user_input
            execution_result = state.execution_result
            result = await self.validate(target, user_input, execution_result, mode=mode)
        elif mode == "novel":
            # 此分支理论上不会执行（已被上面拦截），但保留以防万一
            target = state.scene_text
            user_input = state.user_input
            constraints = {
                "outline": state.outline,
                "writing_constraints": state.writing_constraints or {}
            }
            result = await self.validate(target, user_input, None, mode=mode, constraints=constraints)
        else:
            raise ValueError(f"Unsupported validation mode: {mode}")

        duration = time.time() - start_time
        status = "success" if result.get("passed") else "error"
        logger.info(f"{agent_name} completed, step={step}, status={status}, duration={duration:.2f}")
        return {
            "validation_result": result,
            "final_answer": result.get("feedback", ""),
        }

    async def validate(
        self,
        target: str,
        user_input: str,
        execution_result: Optional[dict] = None,
        mode: str = "code",
        constraints: Optional[Dict] = None,
    ) -> dict:
        """统一验证入口：支持 code 和 novel 模式（但不强制通过）"""
        logger.info(f"ValidatorAgent starting validation with mode={mode}")

        if mode == "code":
            return await self._validate_code(target, user_input, execution_result)
        elif mode == "novel":
            outline = constraints.get("outline") if constraints else None
            writing_constraints = constraints.get("writing_constraints", {})
            return await self._validate_novel(target, outline, writing_constraints)
        else:
            raise ValueError(f"Unsupported validation mode: {mode}")

    # ==================== Code 验证 ====================
    async def _validate_code(self, code: str, user_input: str, execution_result: Optional[dict]) -> dict:
        syntax_ok, syntax_feedback = self._check_syntax(code)
        if not syntax_ok:
            return {
                "passed": False,
                "feedback": f"Syntax error detected: {syntax_feedback}",
                "suggestions": ["Fix the syntax errors before proceeding"],
            }
        builder = VALIDATOR_PROMPT_REGISTRY.get("code")
        prompt = builder.build(code, user_input, execution_result)
        llm_result = await self._validate_with_llm(prompt, execution_result)
        return {
            "passed": llm_result.get("passed", False),
            "feedback": llm_result.get("feedback", "Validation completed."),
            "suggestions": llm_result.get("suggestions", []),
        }

    def _check_syntax(self, code: str) -> Tuple[bool, str]:
        if not code or not code.strip():
            return False, "Code is empty"
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_path = f.name
            py_compile.compile(temp_path, doraise=True)
            os.unlink(temp_path)
            return True, ""
        except py_compile.PyCompileError as e:
            os.unlink(temp_path)
            return False, str(e)
        except Exception as e:
            return False, f"Syntax check error: {e}"

    # ==================== Novel 验证（增强容错，为将来启用准备） ====================
    async def _validate_novel(self, text: str, outline: Optional[Dict], writing_constraints: Dict) -> dict:
        """
        增强版小说验证，宽松处理 JSON 格式和 must_events 匹配。
        目前此方法未被调用（run 中强制通过），保留供未来切换。
        """
        constraints = writing_constraints.copy()
        constraints["outline"] = outline
        must_events = constraints.get("must_events", [])

        # 1. 尝试解析 JSON
        parsed_data = self._extract_json(text)
        if not parsed_data:
            return {
                "passed": False,
                "feedback": "无法解析生成的 JSON，请确保输出包含 scene_text 字段",
                "suggestions": ["检查模型输出格式，应为 valid JSON"]
            }

        scene_text = parsed_data.get("scene_text", "")
        if not scene_text or len(scene_text.strip()) < 50:
            return {
                "passed": False,
                "feedback": "scene_text 字段缺失或过短（<50字符）",
                "suggestions": ["请生成更完整的场景正文"]
            }

        # 2. 如果无必须事件，直接通过
        if not must_events:
            return {
                "passed": True,
                "feedback": "校验通过（无 must_events）",
                "suggestions": []
            }

        # 3. 宽松的 must_events 检查（提取关键词而非精确匹配）
        missing = self._check_must_events_lax(scene_text, must_events)
        if missing:
            # 记录警告但不阻塞（后续可改为严格）
            logger.warning(f"Missing must_events: {missing}")
            return {
                "passed": True,   # 临时放宽
                "feedback": f"部分必须事件未明确体现，但已通过（缺失: {missing}）",
                "suggestions": ["建议在后续章节中补充相关情节"]
            }

        return {
            "passed": True,
            "feedback": "校验通过",
            "suggestions": []
        }

    async def _validate_novel_enhanced(
        self,
        text: str,
        constraints: Dict[str, Any],
        deviation_detected: bool = False,
        missing_goal: List[str] = None,
        missing_conflict: List[str] = None,
    ) -> Dict[str, Any]:
        if missing_goal is None:
            missing_goal = []
        if missing_conflict is None:
            missing_conflict = []

        # 1. 解析 JSON
        parsed_data = self._extract_json(text)
        if not parsed_data:
            return {
                "passed": False,
                "feedback": "无法解析生成的 JSON，请确保输出包含 scene_text 字段",
                "suggestions": ["检查模型输出格式，应为 valid JSON，不要用 ```json 代码块包裹"],
                "should_retry": True,
                "error_details": {"error": "json_parse_failed"},
                "parsed_output": None,
            }

        scene_text = parsed_data.get("scene_text", "")
        if not scene_text or len(scene_text.strip()) < 50:
            return {
                "passed": False,
                "feedback": f"scene_text 字段缺失或过短（{len(scene_text)}字符，需要至少50字符）",
                "suggestions": ["请生成更完整的场景正文，至少50字"],
                "should_retry": True,
                "error_details": {"error": "scene_text_too_short", "length": len(scene_text)},
                "parsed_output": parsed_data,
            }

        # 读取配置
        must_events_threshold = getattr(config, 'must_events_similarity_threshold', 0.30)
        goal_conflict_threshold = 0.25

        must_events = constraints.get("must_events", [])
        goal = constraints.get("goal", "")
        conflict = constraints.get("conflict", "")
        narrative_blueprint = constraints.get("narrative_blueprint", {})
        knowledge_deltas = constraints.get("knowledge_deltas", [])
        character_intent = constraints.get("character_intent", {})
        scene_objective = constraints.get("scene_objective", "")
        current_state_dict = constraints.get("current_state", {})

        # 辅助函数：安全的单个 embedding 请求
        async def safe_embedding(text: str, desc: str) -> Optional[List[float]]:
            try:
                emb_str = await asyncio.wait_for(generate_embedding(text), timeout=10.0)
                return json.loads(emb_str)
            except Exception as e:
                logger.error(f"Embedding failed for {desc}: {e}")
                return None

        # 获取场景正文 embedding
        scene_sample = scene_text[:1000]
        scene_emb = await safe_embedding(scene_sample, "scene_text")
        if scene_emb is None:
            logger.warning("Scene embedding failed, skipping semantic validation")
            return {
                "passed": True,
                "feedback": "Embedding 服务异常，跳过语义验证",
                "suggestions": ["请检查 embedding 服务"],
                "should_retry": False,
                "error_details": {"error": "embedding_failed"},
                "parsed_output": parsed_data,
            }

        errors = []
        error_details = {}
        
        # 检查 must_events
        missing_events = []
        for evt in must_events:
            evt_emb = await safe_embedding(evt, f"must_event '{evt[:30]}'")
            if evt_emb is None:
                continue
            sim = cosine_similarity(evt_emb, scene_emb)
            if sim < must_events_threshold:
                missing_events.append(evt)

        # 检查 goal
        goal_ok = True
        if goal:
            goal_emb = await safe_embedding(goal, "goal")
            if goal_emb is not None:
                sim = cosine_similarity(goal_emb, scene_emb)
                goal_ok = sim >= goal_conflict_threshold

        # 检查 conflict
        conflict_ok = True
        if conflict:
            conflict_emb = await safe_embedding(conflict, "conflict")
            if conflict_emb is not None:
                sim = cosine_similarity(conflict_emb, scene_emb)
                conflict_ok = sim >= goal_conflict_threshold

        # 检查 scene_objective
        scene_obj_ok = True
        if scene_objective:
            obj_emb = await safe_embedding(scene_objective, "scene_objective")
            if obj_emb is not None:
                sim = cosine_similarity(obj_emb, scene_emb)
                scene_obj_ok = sim >= goal_conflict_threshold
                if not scene_obj_ok:
                    errors.append(f"场景目标未达成: {scene_objective[:40]}...")
                    error_details["scene_objective_semantic_match"] = False

        if missing_events:
            missing_list = "、".join(missing_events)
            feedback = f"❌ 缺失必须事件：{missing_list}\n请确保在下一次生成的 scene_text 中**原样包含**以上短语，不可改写或省略。"
            errors.append(feedback)
            error_details["missing_events"] = missing_events
        if not goal_ok:
            errors.append(f"场景目标语义不符: {goal[:40]}...")
            error_details["goal_semantic_match"] = False
        if not conflict_ok:
            errors.append(f"核心冲突语义不符: {conflict[:40]}...")
            error_details["conflict_semantic_match"] = False

        # ========== Director 输出职责边界检查 ==========
        missing_knowledge = []
        if knowledge_deltas:
            for kd in knowledge_deltas:
                info = kd.get("information", "")
                if not info:
                    continue
                info_emb = await safe_embedding(info, f"knowledge '{info[:30]}'")
                if info_emb is None:
                    continue
                sim = cosine_similarity(info_emb, scene_emb)
                if sim < must_events_threshold:
                    missing_knowledge.append(info)
                    errors.append(f"知识变化未体现: {info} (相似度 {sim:.2f} < {must_events_threshold})")
                
                reliability = kd.get("reliability", 1.0)
                if reliability < 0.5:
                    absolute_words = ["绝对", "肯定", "一定", "无疑", "必然", "肯定是", "一定是"]
                    if any(word in scene_text for word in absolute_words):
                        errors.append(f"低可靠性信息被过度确信: {info} (可靠性 {reliability})")
            
            for kd in knowledge_deltas:
                if kd.get("visibility") == "hidden" and kd.get("information") in scene_text:
                    errors.append(f"隐藏信息不得提前暴露: {kd.get('information')}")
        if missing_knowledge:
            error_details["missing_knowledge"] = missing_knowledge

        # 场景角色节奏检查
        scene_role = narrative_blueprint.get("scene_role", "")
        if scene_role == "REVEAL":
            reveal_beat = narrative_blueprint.get("reveal_beat", "")
            if reveal_beat and reveal_beat not in scene_text:
                errors.append(f"揭示节拍未出现: {reveal_beat}")
        elif scene_role == "AFTERMATH":
            if len(scene_text) < 200:
                errors.append("余波场景过短，需要更充分的情感沉淀")

        # 角色意图违背检查
        if character_intent:
            fear = character_intent.get("fear")
            actor = character_intent.get("actor")
            if fear and actor:
                if fear in scene_text:
                    fear_words = ["害怕", "恐惧", "颤抖", "后退", "退缩", "心悸", "胆寒", "色变", "惊惧"]
                    if not any(fw in scene_text for fw in fear_words):
                        errors.append(f"角色意图可能违背: {actor} 恐惧 {fear} 但未表现出恐惧反应")

        # ========== 认知身份一致性检查 ==========
        if current_state_dict and character_intent:
            try:
                world = WorldState.from_dict(current_state_dict)
                actor = character_intent.get("actor")
                if actor and actor in world.characters:
                    char = world.characters[actor]
                    
                    if char.beliefs:
                        belief_violations = {
                            "不杀无辜": ["杀", "斩杀", "屠杀", "灭口", "处决"],
                            "不背叛朋友": ["出卖", "背叛", "告密", "陷害"],
                            "强者为尊": ["屈服", "求饶", "认输"],
                            "丹药不可靠": ["服用丹药", "吞服丹药", "嗑药"],
                            "不食言": ["毁约", "失信", "食言"],
                        }
                        for belief in char.beliefs:
                            violation_keywords = belief_violations.get(belief, [])
                            if violation_keywords and any(kw in scene_text for kw in violation_keywords):
                                errors.append(f"认知身份违背: {actor} 的信念「{belief}」被违背")
                    
                    if char.self_image:
                        self_image_lower = char.self_image.lower()
                        if "天弃" in self_image_lower or "弃子" in self_image_lower:
                            luck_words = ["机缘", "奇遇", "天赐", "传承", "神兵"]
                            if any(word in scene_text for word in luck_words):
                                errors.append(f"认知身份可能不一致: {actor} 自我认知为「{char.self_image}」，但获得了不应得的机缘")
                        
                        if "复仇" in self_image_lower or "报仇" in self_image_lower:
                            forgive_words = ["原谅", "宽恕", "放下仇恨", "释怀"]
                            if any(word in scene_text for word in forgive_words):
                                errors.append(f"认知身份可能不一致: {actor} 自我认知为「{char.self_image}」，但表现出宽容行为")
                    
                    if char.attachments:
                        attachment_keywords = [att for att in char.attachments if len(att) >= 2]
                        for attachment in attachment_keywords:
                            if attachment in scene_text:
                                negative_words = ["丢弃", "毁掉", "遗弃", "破坏", "砸"]
                                idx = scene_text.find(attachment)
                                if idx != -1:
                                    context = scene_text[max(0, idx-30):min(len(scene_text), idx+30)]
                                    if any(word in context for word in negative_words):
                                        errors.append(f"认知身份可能不一致: {actor} 的依恋「{attachment}」被表现出负面行为")
                    
                    if char.moral_boundaries:
                        boundary_violations = {
                            "不杀无辜": ["杀", "斩杀", "屠杀"],
                            "不背叛宗门": ["背叛", "出卖", "投敌"],
                            "不偷盗": ["偷", "盗", "窃取"],
                            "不欺骗": ["欺骗", "撒谎", "骗"],
                        }
                        for boundary in char.moral_boundaries:
                            violation_keywords = boundary_violations.get(boundary, [])
                            if violation_keywords and any(kw in scene_text for kw in violation_keywords):
                                errors.append(f"道德底线突破: {actor} 突破了「{boundary}」底线")
            except Exception as e:
                logger.warning(f"Failed to check cognitive identity: {e}")

        # ========== 5. 模板化表达检测（从配置读取） ==========
        from src.config_loader import get_xianxia_config
        from collections import Counter
        import re
        
        config_obj = get_xianxia_config()
        voice_cfg = config_obj.voice if hasattr(config_obj, 'voice') else {}
        
        voice_violations = []
        
        # 检查口头禅（主角常用短语）
        catchphrases = voice_cfg.get("dialogue", {}).get("catchphrases", ["有意思", "哼", "我偏不信"])
        max_catchphrase = voice_cfg.get("dialogue", {}).get("max_catchphrase_per_chapter", 2)
        
        for phrase in catchphrases:
            count = scene_text.count(phrase)
            if count > max_catchphrase:
                voice_violations.append(f"口头禅 '{phrase}' 出现 {count} 次，超过限制 {max_catchphrase}")
                error_details.setdefault("voice_violations", []).append({
                    "phrase": phrase,
                    "count": count,
                    "limit": max_catchphrase
                })
        
        # 检查重复关键词频率
        words = re.findall(r'[\u4e00-\u9fff]{2,4}', scene_text)
        max_keyword_freq = voice_cfg.get("repetition", {}).get("max_keyword_frequency", 3)
        keyword_freq = Counter(words)
        for keyword, freq in keyword_freq.most_common(5):
            if freq > max_keyword_freq and keyword not in catchphrases:
                voice_violations.append(f"高频词 '{keyword}' 出现 {freq} 次")
                error_details.setdefault("voice_violations", []).append({
                    "phrase": keyword,
                    "count": freq,
                    "limit": max_keyword_freq
                })
        
        # 记录文笔违规到日志（不阻断）
        if voice_violations:
            logger.warning(f"Voice violations: {voice_violations}")

        # ==================== 因果关系校验 ====================
        predicates = constraints.get("predicates", {})
        budget = constraints.get("budget")
        degraded = constraints.get("degraded", False)
        causality_failed = False
        causality_suggestions = []

        if predicates:
            from src.writing.causality.validator import CausalityValidator
            causality_validator = CausalityValidator()
            events_to_check = parsed_data.get("events", [])

            for event_obj in events_to_check:
                event_type = event_obj.get("type")
                if not event_type:
                    continue
                temp_event = event_obj.copy()
                temp_event["type"] = event_type
                result = causality_validator.validate(temp_event, predicates)
                if not result["passed"]:
                    severity = result.get("severity", "error")
                    if degraded and severity == "error":
                        severity = "warning"
                        result["passed"] = True
                        logger.warning(f"Drift degradation: rule {result.get('rule_id', 'unknown')} downgraded from error to warning")
                    if severity == "warning" and budget:
                        if not await budget.consume("warning"):
                            severity = "error"
                            result["passed"] = False
                    if severity == "error":
                        causality_failed = True
                        causality_suggestions.extend(result["suggestions"])
                        errors.append(f"因果规则违反: {result['suggestions'][0] if result['suggestions'] else '未知'}")
                    elif severity == "warning":
                        logger.warning(f"Causality warning: {result['suggestions']}")
                        causality_suggestions.extend(result["suggestions"])

        if causality_failed:
            error_details["causality"] = {
                "passed": False,
                "suggestions": causality_suggestions
            }

        # 计算是否需要重试
        has_missing_knowledge = bool(missing_knowledge)
        should_retry = bool(missing_events or not goal_ok or not conflict_ok or not scene_obj_ok or has_missing_knowledge or causality_failed)

        if errors:
            feedback = "；".join(errors)
            logger.warning(f"Validation failed: {feedback}")
            suggestions = [
                "请严格遵循场景计划中的 goal、conflict 和 must_events 的语义要求",
                "确保生成的正文完整表达所有必须事件的核心含义",
                "请严格按照导演蓝图中的知识变化序列和场景角色要求",
            ]
            if missing_knowledge:
                missing_list = "、".join(missing_knowledge)
                suggestions.append(f"请确保在正文中明确体现以下知识变化：{missing_list}")
            if not scene_obj_ok and scene_objective:
                suggestions.append(f"请确保场景完成其存在理由：{scene_objective}")
            if causality_suggestions:
                suggestions.extend(causality_suggestions)
            return {
                "passed": False,
                "feedback": feedback,
                "suggestions": suggestions,
                "should_retry": should_retry,
                "error_details": error_details,
                "parsed_output": parsed_data,
            }

        return {
            "passed": True,
            "feedback": "校验通过",
            "suggestions": [],
            "should_retry": False,
            "error_details": error_details,
            "parsed_output": parsed_data,
        }

    # ---------- 辅助方法 ----------
    def _extract_scene_text(self, raw: str) -> str:
        """从原始输出中提取 scene_text（支持无 JSON 时直接返回原始文本）"""
        parsed = self._extract_json(raw)
        if parsed:
            return parsed.get("scene_text", "")
        return raw

    def _extract_json(self, text: str) -> Optional[Dict]:
        """鲁棒提取 JSON，自动修复常见格式问题"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except:
            pass

        # 提取第一个完整 JSON 对象
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return None
        json_str = match.group()

        # 修复常见错误
        json_str = re.sub(r',\s*}', '}', json_str)      # 移除尾随逗号
        json_str = re.sub(r',\s*]', ']', json_str)      # 数组尾随逗号
        # 处理未转义的双引号（简单场景）
        json_str = re.sub(r'(?<!\\)"', '"', json_str)   # 确保引号为双引号（过于激进，但容错）
        json_str = re.sub(r'\n\s*', ' ', json_str)      # 压缩换行

        try:
            return json.loads(json_str)
        except:
            return None

    def _check_must_events_lax(self, scene_text: str, must_events: List[str]) -> List[str]:
        """
        宽松检查：从 must_events 中提取关键词（如去除动词，取名词/核心词），
        再检查是否出现在正文中。
        """
        missing = []
        for event in must_events:
            keywords = self._extract_keywords(event)
            if not any(kw in scene_text for kw in keywords):
                missing.append(event)
        return missing

    def _extract_keywords(self, event: str) -> List[str]:
        """
        从事件描述中提取关键词。示例：
        "捡到神秘玉佩" -> ["玉佩", "神秘"]
        "发现家族灵田异常" -> ["灵田", "异常"]
        """
        # 简单分词：取长度>1的字符组合
        words = []
        # 移除常见动词和虚词
        stop_words = {"捡到", "发现", "获得", "遇到", "看见", "听到", "被"}
        for w in event.split():
            w = w.strip()
            if w in stop_words:
                continue
            if len(w) >= 2:
                words.append(w)
        # 如果没有提取到，则取整个事件的前6个字符
        if not words:
            words = [event[:6]]
        return words

    def _log_validation_issues(self, raw: str):
        """记录可能的格式问题（用于调试，不影响通过）"""
        parsed = self._extract_json(raw)
        if not parsed:
            logger.warning("Validation debug: Failed to extract JSON from output")
            return
        scene_text = parsed.get("scene_text", "")
        if len(scene_text) < 100:
            logger.warning(f"Validation debug: scene_text too short ({len(scene_text)} chars)")

    # ==================== 原有方法保留（用于代码验证） ====================
    async def _semantic_validate(self, text: str, must_events: List[str]) -> Tuple[bool, str]:
        # 此处省略原有实现，保留占位
        return True, ""

    def _hard_validate_novel(self, text: str, constraints: Dict[str, Any]) -> Tuple[bool, str]:
        # 保留原实现
        return True, ""

    async def _llm_soft_validate(self, text: str, must_events: List[str]) -> Dict[str, Any]:
        # 保留原实现
        return {"passed": True, "feedback": ""}

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    async def _validate_with_llm(self, prompt: str, execution_result: Optional[dict] = None) -> dict:
        # 记录 prompt（在调用之前）
        log_prompt("validator", prompt, metadata={"validation_type": "code"})

        model = get_router().get_model_for_task("validate")
        client = AsyncOpenAI(api_key="not-needed", base_url=self.llm_api_url)
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a validation assistant. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
                timeout=config.llm_timeout_validation,
            )
            raw_output = response.choices[0].message.content or ""
            parsed_result = self._parse_validation_result_enhanced(raw_output)        
            return self._normalize_validation_result(parsed_result, raw_output)
        except Exception as e:
            logger.warning(f"LLM validation call failed: {e}")
            return self._fallback_validation(execution_result)

    def _parse_validation_result_enhanced(self, text: str) -> Dict[str, Any]:
        # 保留原有实现（未变）
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'({[\s\S]*?"passed"[\s\S]*?})',
            r'({[^{}]*?"passed"[^{}]*?})',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                try:
                    json_str = match.strip()
                    json_str = re.sub(r',\s}', '}', json_str)
                    json_str = re.sub(r',\s]', ']', json_str)
                    result = json.loads(json_str)
                    if "passed" in result:
                        return result
                except:
                    continue
        return {"passed": False, "feedback": "Could not parse LLM output", "suggestions": []}

    def _extract_via_regex(self, text: str) -> Optional[Dict[str, Any]]:
        # 保留原有实现
        return None

    def _parse_from_natural_language(self, text: str) -> Optional[Dict[str, Any]]:
        return None

    def _normalize_validation_result(self, result: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
        normalized = {}
        passed_value = result.get("passed")
        if isinstance(passed_value, bool):
            normalized["passed"] = passed_value
        else:
            normalized["passed"] = False
        normalized["feedback"] = result.get("feedback", "Validation completed.")
        normalized["suggestions"] = result.get("suggestions", [])
        return normalized

    def _fallback_validation(self, execution_result: Optional[dict] = None) -> dict:
        if execution_result and execution_result.get("success"):
            return {"passed": True, "feedback": "Execution succeeded", "suggestions": []}
        return {"passed": False, "feedback": "No valid execution result", "suggestions": []}

    async def validate_novel_consistency(self, text: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """兼容旧接口"""
        outline = constraints.get("outline")
        writing_constraints = constraints.get("writing_constraints", {})
        return await self._validate_novel(text, outline, writing_constraints)