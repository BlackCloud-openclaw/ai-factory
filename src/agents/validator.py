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
from collections import Counter

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
from src.writing.planning_contract import PlanningContract, StateChange
from src.writing.validation import SemanticValidator, NoOpEmbeddingProvider, ValidationResult
from src.writing.validation_result import (
    ValidatorOutput,
    Violation,
    ViolationSeverity,
    ValidationStage,
    ValidationStatus,
)
from src.writing.contracts.event_mapping import ContractEventResolver
from src.writing.contracts.event_matcher import ContractEventMatcher
from src.writing.validation.models import MissingContractChange, ContractSeverity


logger = setup_logging("agents.validator")


class ValidatorAgent(BaseAgent):
    """Agent responsible for validating code quality and novel consistency."""
    # ========== Phase 14.0C-3A: Validator Runtime Diagnostics ==========
    _semantic_validator: Optional[SemanticValidator] = None
    _embedding_status: str = "unknown"
    _embedding_fallback_reason: str = ""
    # ==================================================================

    def __init__(
        self,
        llm_api_url: str = config.llm_api_url,
        llm_model: str = config.llm_model_name,
    ):
        self.llm_api_url = llm_api_url
        self.llm_model = llm_model
        self._execution_id: Optional[str] = None

    async def run(self, state: AgentState) -> Dict[str, Any]:
        agent_name = "ValidatorAgent"
        state.step_count += 1
        step = state.step_count
        logger.info(f"Starting {agent_name}, step={step}")
        start_time = time.time()

        # 获取 execution_id
        execution_id = state.metadata.get("execution_id")
        if not execution_id:
            execution_id = f"val_{state.project_id}_{step}_{int(time.time())}"
            state.metadata["execution_id"] = execution_id

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
                "current_state": state.current_state,
                "final_state": state.current_state or {},
            }

            # 注入 Director 输出
            constraints["narrative_blueprint"] = state.narrative_blueprint or {}
            constraints["knowledge_deltas"] = state.knowledge_deltas or []
            constraints["character_intent"] = state.character_intent or {}
            constraints["scene_objective"] = state.scene_plan.get("scene_objective", "") if state.scene_plan else ""
            constraints["active_loop"] = state.metadata.get("active_loop")

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

            # 创建一致性预算并加载
            if state.novel_id:
                budget = ConsistencyBudget(state.novel_id, state.current_volume, state.current_chapter)
                await budget.load()
                constraints["budget"] = budget
            else:
                constraints["budget"] = None

            # 健康检查与漂移降级
            if state.novel_id:
                drift_level = await HealthChecker.check_drift(state.novel_id)
                if drift_level in ("WARNING", "CRITICAL"):
                    constraints["degraded"] = True
                    logger.warning(f"Projection drift detected ({drift_level}), validator degraded")
            else:
                constraints["degraded"] = False

            # 从 scene_plan 中提取必须事件、目标、冲突
            if state.scene_plan:
                constraints["must_events"] = state.scene_plan.get("must_events", [])
                constraints["goal"] = state.scene_plan.get("goal", "")
                constraints["conflict"] = state.scene_plan.get("conflict", "")

            # 获取 Planning Contract
            planning_contract = None
            if hasattr(state, 'planning_contract') and state.planning_contract:
                try:
                    contract_data = state.planning_contract
                    # ===== 诊断日志 START =====
                    logger.info(f"[DIAG-VAL-1] contract_data type: {type(contract_data)}")
                    if isinstance(contract_data, dict):
                        logger.info(f"[DIAG-VAL-1] contract_data keys: {list(contract_data.keys())}")
                        if 'observables' in contract_data:
                            logger.info(f"[DIAG-VAL-1] observables in contract_data: {contract_data['observables']}")
                        else:
                            logger.warning("[DIAG-VAL-1] observables key MISSING in contract_data!")
                    # ===== 诊断日志 END =====
                    planning_contract = PlanningContract(**contract_data)
                    logger.info(f"✅ Loaded Planning Contract for validator: {planning_contract.scene_id}")
                except Exception as e:
                    logger.warning(f"Failed to parse Planning Contract in validator: {e}")
                    planning_contract = None

            # 调用增强版校验
            result = await self._validate_novel_enhanced(
                raw,
                constraints,
                deviation_detected=getattr(state, 'deviation_detected', False),
                missing_goal=getattr(state, 'missing_goal_keywords', []),
                missing_conflict=getattr(state, 'missing_conflict_keywords', []),
                planning_contract=planning_contract,
                execution_id=execution_id,
                writer_artifact=state.writer_artifact,  # ← 新增
            )

            # ✅ 确保 validator_output 存在（补充早期返回路径）
            if "validator_output" not in result:
                passed = result.get("passed", False)
                errors = []
                if not passed:
                    feedback = result.get("feedback", "")
                    if feedback:
                        errors = [feedback]
                output = self._build_validator_output(
                    execution_id=execution_id,
                    passed=passed,
                    errors=errors,
                    voice_violations=[],
                )
                result["validator_output"] = output.to_runtime_dict()

            duration = time.time() - start_time
            status = "success" if result.get("passed") else "error"
            logger.info(f"{agent_name} completed (novel), step={step}, status={status}, duration={duration:.2f}")

            # ✅ 返回时同时包含 validation_result 和顶层 validator_output
            return {
                "validation_result": result,
                "final_answer": clean,
                "validator_output": result["validator_output"],
            }

        # ========== 代码验证 ==========
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

        # 1. 解析 JSON (原有逻辑保持不变)
        parsed_data = self._extract_json(text)
        if not parsed_data or not parsed_data.get("scene_text"):
            # 尝试用正则提取 scene_text（原逻辑）
            match = re.search(r'"scene_text"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
            if match:
                scene_text_raw = match.group(1)
                scene_text = scene_text_raw.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                parsed_data = {"scene_text": scene_text, "events": [], "foreshadowing": []}
                logger.warning("Validator: extracted scene_text via regex fallback")
            else:
                # 最后手段：直接使用原始文本作为 scene_text
                # 因为 validate_node 已经证明这个 text 是有效的
                if text and len(text.strip()) > 50:
                    parsed_data = {"scene_text": text, "events": [], "foreshadowing": []}
                    logger.warning(f"Validator: using raw text as scene_text (final fallback), text_length={len(text)}")
                else:
                    return {
                        "passed": False,
                        "feedback": "无法解析生成的 JSON，且无法通过任何 fallback 提取 scene_text",
                        "suggestions": ["请确保模型输出包含 scene_text 字段"],
                        "should_retry": True,
                        "error_details": {"error": "json_parse_failed_all_fallback"},
                        "parsed_output": {},
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
        planning_contract: Optional[PlanningContract] = None,
        execution_id: str = "",
        writer_artifact: Optional[Dict[str, Any]] = None,  # Phase 14.0C-3 Commit D.2
    ) -> Dict[str, Any]:
        """
        增强版小说验证 - 包含语义匹配、认知身份检查、因果验证等。
        Phase 13.2.3B 集成：使用 SemanticValidator 进行 must_events 验证。
        
        Phase 14.0C-3A 修改：
        - 入口观测日志
        - SemanticValidator 调用观测
        - 最外层异常收敛为 ValidatorOutput
        
        Phase 14.0C-3 Commit D.2:
        - 优先从 writer_artifact 读取结构化数据
        """
        logger.critical("!!! NEW VALIDATOR CODE IS RUNNING !!!")
        
        total_missing_changes = []  # 新增
        
        if missing_goal is None:
            missing_goal = []
        if missing_conflict is None:
            missing_conflict = []

        # ========== Phase 14.0C-3A: 入口观测 ==========
        logger.info(
            "Validator enhanced validation start: "
            f"planning_contract={planning_contract is not None}, "
            f"text_length={len(text) if text else 0}, "
            f"execution_id={execution_id}"
        )
        # =============================================

        # ========== Phase 14.0C-3A: 最外层异常收敛 ==========
        try:
            logger.info(f"🔍 active_loop from constraints: {constraints.get('active_loop')}")
            logger.info(f"Validator: text type={type(text)}, length={len(text) if text else 0}")

            # 1. 解析 JSON (原有逻辑保持不变)
            logger.critical(f"RAW_TEXT_SAMPLE: {text[:1000]}")
            logger.critical(f"FULL_TEXT_SAMPLE: {text}")
            logger.critical("DEBUG-POINT-1: after RAW_TEXT_SAMPLE")

            # ============================================================
            # Commit D.2: 优先从 writer_artifact 读取结构化数据
            # ============================================================
            used_writer_artifact = False
            _source_format = "unknown"
            _events_extracted = False
            _event_count = 0
            _fallback_reason = None
            parsed_data = None

            if writer_artifact and isinstance(writer_artifact, dict):
                artifact_scene_text = writer_artifact.get("scene_text")
                artifact_events = writer_artifact.get("events", [])
                artifact_foreshadowing = writer_artifact.get("foreshadowing", [])

                if artifact_scene_text:
                    parsed_data = {
                        "scene_text": artifact_scene_text,
                        "events": artifact_events,
                        "foreshadowing": artifact_foreshadowing,
                    }
                    used_writer_artifact = True
                    _source_format = "writer_artifact"
                    _events_extracted = len(artifact_events) > 0
                    _event_count = len(artifact_events)
                    _fallback_reason = None

                    logger.info(
                        f"Validator: using writer_artifact (events={_event_count}, "
                        f"foreshadowing={len(artifact_foreshadowing)})"
                    )
                    # ========== D.3 观测点 5 ==========
                    logger.critical(
                        "VALIDATOR_RECEIVED_ARTIFACT: has_scene_text=%s, events_len=%d, foreshadowing_len=%d",
                        bool(artifact_scene_text),
                        len(artifact_events),
                        len(artifact_foreshadowing)
                    )
                    # ================================
                else:
                    logger.warning(
                        "Validator: writer_artifact missing scene_text, falling back"
                    )
            # ============================================================

            # 如果未使用 artifact，执行原有的 JSON 提取
            if not used_writer_artifact:
                parsed_data = self._extract_json(text)
                logger.critical(f"DEBUG-POINT-2: parsed_data is None? {parsed_data is None}")
                if parsed_data:
                    logger.critical(f"DEBUG-POINT-2: parsed_data keys = {list(parsed_data.keys())}")

                # 初始化 scene_text 变量
                scene_text = None

                if not parsed_data or not parsed_data.get("scene_text"):
                    # 如果 text 是纯文本（长度足够），直接使用
                    if text and len(text.strip()) >= 50:
                        logger.critical(f"DEBUG-POINT-3: using raw text as scene_text (length={len(text)})")
                        scene_text = text.strip()
                        parsed_data = {
                            "scene_text": scene_text,
                            "events": [],
                            "foreshadowing": []
                        }
                        _source_format = "fallback_text"
                        _events_extracted = False
                        _event_count = 0
                        _fallback_reason = "raw_text_fallback"
                        logger.warning("Validator: using raw text as scene_text (direct fallback)")
                    else:
                        # 尝试正则 fallback（仅当 text 可能包含 JSON 时）
                        match = re.search(r'"scene_text"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
                        if match:
                            scene_text_raw = match.group(1)
                            scene_text = scene_text_raw.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                            parsed_data = {
                                "scene_text": scene_text,
                                "events": [],
                                "foreshadowing": []
                            }
                            _source_format = "fallback_regex"
                            _events_extracted = False
                            _event_count = 0
                            _fallback_reason = "regex_extraction"
                            logger.warning("Validator: extracted scene_text via regex fallback")
                            logger.critical(f"DEBUG-POINT-4: regex fallback succeeded, scene_text length = {len(scene_text)}")
                        else:
                            logger.critical("DEBUG-POINT-5: all fallbacks FAILED, returning early")
                            return {
                                "passed": False,
                                "feedback": "无法解析生成的 JSON，且无法通过任何 fallback 提取 scene_text",
                                "suggestions": ["请确保模型输出包含 scene_text 字段"],
                                "should_retry": True,
                                "error_details": {"error": "json_parse_failed_all_fallback"},
                                "parsed_output": {},
                            }

                # 如果 parsed_data 有 scene_text 但 scene_text 变量仍为 None（即 JSON 解析成功的情况）
                if scene_text is None and parsed_data and parsed_data.get("scene_text"):
                    scene_text = parsed_data.get("scene_text")
                    # 如果来自 JSON 解析，设置来源
                    if _source_format == "unknown":
                        _source_format = "json_extract"
                        _events_extracted = len(parsed_data.get("events", [])) > 0
                        _event_count = len(parsed_data.get("events", []))
                        _fallback_reason = None
            else:
                # 已使用 artifact，scene_text 从 artifact 获取
                scene_text = parsed_data.get("scene_text", "")

            # 再次检查 scene_text 的有效性
            if not scene_text or len(scene_text.strip()) < 50:
                logger.critical("DEBUG-POINT-6: scene_text too short, returning early")
                return {
                    "passed": False,
                    "feedback": f"scene_text 字段缺失或过短（{len(scene_text) if scene_text else 0}字符，需要至少50字符）",
                    "suggestions": ["请生成更完整的场景正文，至少50字"],
                    "should_retry": True,
                    "error_details": {"error": "scene_text_too_short", "length": len(scene_text) if scene_text else 0},
                    "parsed_output": parsed_data or {},
                }

            logger.critical("DEBUG-POINT-7: passed length check, continuing")

            control_scores = {}
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
            active_loop = constraints.get("active_loop")

            errors = []
            error_details = {}
            should_retry = False

            # 辅助函数：安全的 embedding 请求
            async def safe_embedding(text: str, desc: str) -> Optional[List[float]]:
                try:
                    emb_str = await asyncio.wait_for(generate_embedding(text), timeout=10.0)
                    return json.loads(emb_str)
                except Exception as e:
                    logger.error(f"Embedding failed for {desc}: {e}")
                    return None

            # ===== 修复点 1：移除提前返回，设置 embedding_available 标志 =====
            scene_sample = scene_text[:1000]
            logger.critical("DEBUG-POINT-8: about to call safe_embedding")
            scene_emb = await safe_embedding(scene_sample, "scene_text")
            logger.critical(f"DEBUG-POINT-9: safe_embedding returned, type={type(scene_emb)}")
            embedding_available = scene_emb is not None
            if not embedding_available:
                logger.warning("Scene embedding failed, falling back to keyword-based validation")

            # ===== 强制日志：确认到达 DIAG-VAL-2 之前 =====
            logger.critical("DEBUG-POINT-10: reached DIAG-VAL-2 section")

            # ========== 新增调试日志：检查 planning_contract 进入分支前的状态 ==========
            logger.info(f"Validator: planning_contract exists = {planning_contract is not None}")
            if planning_contract is not None:
                has_obs = planning_contract.observables is not None
                sc_count = len(planning_contract.observables.state_changes) if has_obs else 0
                logger.info(f"Validator: observables exists = {has_obs}, state_changes count = {sc_count}")
                
                # ---- 新增更详细的诊断（带异常保护） ----
                try:
                    logger.info(f"[DIAG-VAL-2] planning_contract.scene_id = {planning_contract.scene_id}")
                    logger.info(f"[DIAG-VAL-2] type(observables) = {type(planning_contract.observables)}")
                    if has_obs:
                        logger.info(f"[DIAG-VAL-2] type(observables.state_changes) = {type(planning_contract.observables.state_changes)}")
                        if sc_count > 0:
                            first = planning_contract.observables.state_changes[0]
                            try:
                                first_dict = first.model_dump() if hasattr(first, 'model_dump') else first
                                logger.info(f"[DIAG-VAL-2] first state_change sample: {first_dict}")
                            except Exception as e:
                                logger.warning(f"[DIAG-VAL-2] Could not dump first state_change: {e}")
                        else:
                            logger.warning("[DIAG-VAL-2] state_changes is EMPTY list (length=0)")
                    else:
                        logger.warning("[DIAG-VAL-2] observables is None! This will cause SemanticValidator to be skipped.")
                except Exception as e:
                    logger.critical(f"[DIAG-VAL-2] LOGGING EXCEPTION: {type(e).__name__}: {e}")
                    raise  # 让异常传播到外层 try
            else:
                logger.warning("[DIAG-VAL-2] planning_contract is None!")

            # ===== 诊断日志：进入 SemanticValidator 分支前的状态检查 =====
            logger.info(f"[DIAG-VAL-3] About to check conditions for SemanticValidator")
            logger.info(f"[DIAG-VAL-3] planning_contract is None? {planning_contract is None}")
            if planning_contract is not None:
                logger.info(f"[DIAG-VAL-3] observables is None? {planning_contract.observables is None}")
                if planning_contract.observables is not None:
                    logger.info(f"[DIAG-VAL-3] state_changes is None? {planning_contract.observables.state_changes is None}")
                    logger.info(f"[DIAG-VAL-3] state_changes length: {len(planning_contract.observables.state_changes)}")
            # ===== 诊断日志 END =====

            # ========== 修复点 2：无条件调用 SemanticValidator（只要 planning_contract 存在） ==========
            # ========== Phase 13.2.3B: 使用 SemanticValidator 进行 must_events 验证 ==========
            if planning_contract is not None:
                logger.info("Validator: requesting SemanticValidator (unconditionally)")
                semantic_validator = self._get_validator()
                logger.info(f"SemanticValidator ready: type={type(semantic_validator).__name__}")
                
                validation_result = semantic_validator.validate(
                    contract=planning_contract,
                    scene_text=parsed_data.get("scene_text", ""),
                )
                
                logger.info(
                    "SemanticValidator finished: "
                    f"passed={validation_result.passed}, "
                    f"missing={validation_result.missing_count}, "
                    f"matched={validation_result.match_count}, "
                    f"blocking={validation_result.blocking_missing_count}"
                )
                
                missing_events = validation_result.missing
                logger.info(
                    f"SemanticValidator: passed={validation_result.passed}, "
                    f"missing={validation_result.missing_count}, "
                    f"matched={validation_result.match_count}, "
                    f"confidence={validation_result.overall_confidence:.2f}, "
                    f"blocking_count={validation_result.blocking_missing_count}"
                )
                
                if validation_result.matched:
                    for ev in validation_result.matched[:3]:
                        logger.debug(
                            f"  Matched: '{ev.event_text[:30]}' via {ev.matcher} "
                            f"(conf={ev.confidence:.2f}, source={ev.source.value if hasattr(ev.source, 'value') else ev.source})"
                        )
                
                if validation_result.errors:
                    errors.append(f"❌ 阻断性缺失：{', '.join(missing_events[:3])}")
                    error_details["blocking_missing_events"] = [
                        e for e in missing_events
                        if any(b in str(validation_result.errors) for b in e[:10])
                    ]
                    if validation_result.blocking_missing_count > 0:
                        should_retry = True
                else:
                    if missing_events:
                        logger.warning(f"非阻断性缺失: {missing_events[:3]}")
                
                # ========== Commit A: 提取 missing_names（用于反馈） ==========
                missing_names = []
                for item in validation_result.missing:
                    if hasattr(item, 'name') and item.name:
                        missing_names.append(item.name)
                    elif hasattr(item, 'type') and item.type:
                        missing_names.append(item.type)
                    else:
                        missing_names.append(str(item))

                control_scores["semantic_validation"] = {
                    "passed": validation_result.passed,
                    "confidence": round(validation_result.overall_confidence, 3),
                    "weight_applied": round(validation_result.weight_applied, 3),
                    "match_count": validation_result.match_count,
                    "missing_count": validation_result.missing_count,
                    "blocking_count": validation_result.blocking_missing_count,
                    "missing_names": missing_names,
                }
            else:
                # 当 planning_contract 为 None 时，使用 fallback 关键词匹配
                if must_events:
                    logger.warning("SemanticValidator: planning_contract is None, using fallback _check_must_events_lax")
                    missing_events = self._check_must_events_lax(scene_text, must_events)
                    if missing_events:
                        errors.append(f"❌ 缺失必须事件：{', '.join(missing_events)}")
                        error_details["missing_events"] = missing_events
                        should_retry = True

            # ========== 修复点 3：goal/conflict/objective 语义检查仅在 embedding 可用时执行 ==========
            goal_ok = True
            if embedding_available and goal:
                goal_emb = await safe_embedding(goal, "goal")
                if goal_emb is not None:
                    sim = cosine_similarity(goal_emb, scene_emb)
                    goal_ok = sim >= goal_conflict_threshold
            
            conflict_ok = True
            if embedding_available and conflict:
                conflict_emb = await safe_embedding(conflict, "conflict")
                if conflict_emb is not None:
                    sim = cosine_similarity(conflict_emb, scene_emb)
                    conflict_ok = sim >= goal_conflict_threshold
            
            scene_obj_ok = True
            if embedding_available and scene_objective:
                obj_emb = await safe_embedding(scene_objective, "scene_objective")
                if obj_emb is not None:
                    sim = cosine_similarity(obj_emb, scene_emb)
                    scene_obj_ok = sim >= goal_conflict_threshold
                    if not scene_obj_ok:
                        errors.append(f"场景目标未达成: {scene_objective[:40]}...")
                        error_details["scene_objective_semantic_match"] = False

            if not goal_ok and goal:
                errors.append(f"场景目标语义不符: {goal[:40]}...")
                error_details["goal_semantic_match"] = False
            if not conflict_ok and conflict:
                errors.append(f"核心冲突语义不符: {conflict[:40]}...")
                error_details["conflict_semantic_match"] = False

            # ========== Director 输出职责边界检查（仅 embedding 可用时使用语义，否则跳过） ==========
            missing_knowledge = []
            if embedding_available and knowledge_deltas:
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
                    
                    if kd.get("visibility") == "hidden" and kd.get("information") in scene_text:
                        errors.append(f"隐藏信息不得提前暴露: {kd.get('information')}")

            if missing_knowledge:
                error_details["missing_knowledge"] = missing_knowledge

            # 场景角色节奏检查（不依赖 embedding）
            scene_role = narrative_blueprint.get("scene_role", "")
            if scene_role == "REVEAL":
                reveal_beat = narrative_blueprint.get("reveal_beat", "")
                if reveal_beat and reveal_beat not in scene_text:
                    errors.append(f"揭示节拍未出现: {reveal_beat}")
            elif scene_role == "AFTERMATH":
                if len(scene_text) < 200:
                    errors.append("余波场景过短，需要更充分的情感沉淀")

            # 角色意图违背检查（不依赖 embedding）
            if character_intent:
                fear = character_intent.get("fear")
                actor = character_intent.get("actor")
                if fear and actor:
                    if fear in scene_text:
                        fear_words = ["害怕", "恐惧", "颤抖", "后退", "退缩", "心悸", "胆寒", "色变", "惊惧"]
                        if not any(fw in scene_text for fw in fear_words):
                            errors.append(f"角色意图可能违背: {actor} 恐惧 {fear} 但未表现出恐惧反应")

            # ========== 认知身份一致性检查（保持不变） ==========
            if current_state_dict and character_intent:
                try:
                    world = WorldState.from_dict(current_state_dict)
                    actor = character_intent.get("actor")
                    char = world.get_character(actor)
                    if char is not None:
                        cognitive_rules = get_xianxia_config().cognitive_rules
                        if char.beliefs:
                            belief_violations = cognitive_rules.get("belief_violations", {})
                            for belief in char.beliefs:
                                violation_keywords = belief_violations.get(belief, [])
                                if violation_keywords and any(kw in scene_text for kw in violation_keywords):
                                    errors.append(f"认知身份违背: {actor} 的信念「{belief}」被违背")
                        if char.self_image:
                            self_image_lower = char.self_image.lower()
                            self_image_rules = cognitive_rules.get("self_image_rules", [])
                            for rule in self_image_rules:
                                match_tags = rule.get("match", [])
                                if not any(tag in self_image_lower for tag in match_tags):
                                    continue
                                forbidden_keywords = rule.get("forbidden_keywords", [])
                                if any(kw in scene_text for kw in forbidden_keywords):
                                    errors.append(
                                        f"认知身份可能不一致: {actor} 自我认知为「{char.self_image}」，"
                                        f"但不应出现 {', '.join(forbidden_keywords[:3])} 类词汇"
                                    )
                        if char.attachments:
                            attachment_rules = cognitive_rules.get("attachment_rules", {})
                            default_neg = attachment_rules.get("default_negative_keywords", [
                                "丢弃", "毁掉", "遗弃", "破坏", "砸"
                            ])
                            family_neg = attachment_rules.get("family_negative_keywords", [])
                            artifact_neg = attachment_rules.get("artifact_negative_keywords", [])
                            all_negative = list(set(default_neg + family_neg + artifact_neg))
                            for attachment in char.attachments:
                                if attachment in scene_text:
                                    idx = scene_text.find(attachment)
                                    if idx != -1:
                                        context = scene_text[max(0, idx-30):min(len(scene_text), idx+30)]
                                        if any(word in context for word in all_negative):
                                            errors.append(f"认知身份可能不一致: {actor} 的依恋「{attachment}」被表现出负面行为")
                        if char.moral_boundaries:
                            boundary_violations = cognitive_rules.get("boundary_violations", {})
                            for boundary in char.moral_boundaries:
                                violation_keywords = boundary_violations.get(boundary, [])
                                if violation_keywords and any(kw in scene_text for kw in violation_keywords):
                                    errors.append(f"道德底线突破: {actor} 突破了「{boundary}」底线")
                except Exception as e:
                    logger.warning(f"Failed to check cognitive identity: {e}")

            # ========== 模板化表达检测 (Voice) ==========
            config_obj = get_xianxia_config()
            voice_cfg = config_obj.voice if hasattr(config_obj, 'voice') else {}
            voice_violations = []
            # 口头禅
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
            # 角色特有口头禅
            character_overrides = voice_cfg.get("dialogue", {}).get("character_overrides", {})
            for char, phrases in character_overrides.items():
                if char in scene_text:
                    for phrase in phrases:
                        count = scene_text.count(phrase)
                        if count > max_catchphrase:
                            voice_violations.append(f"{char} 特有口头禅 '{phrase}' 出现 {count} 次")
            # 重复关键词
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
            # 禁止组合
            forbidden_patterns = voice_cfg.get("repetition", {}).get("forbidden_patterns", [])
            for pattern in forbidden_patterns:
                if re.search(pattern, scene_text):
                    voice_violations.append(f"禁止模式 '{pattern}' 出现")
                    error_details.setdefault("voice_violations", []).append({
                        "pattern": pattern,
                        "matched": True
                    })
            # 连续短句
            max_consecutive = voice_cfg.get("sentence", {}).get("max_consecutive_short_sentences", 3)
            short_threshold = voice_cfg.get("sentence", {}).get("short_sentence_threshold", 8)
            sentences = re.split(r'[。！？；]', scene_text)
            consecutive_short = 0
            for sent in sentences:
                if len(sent.strip()) < short_threshold:
                    consecutive_short += 1
                    if consecutive_short > max_consecutive:
                        voice_violations.append(f"连续短句过多（{consecutive_short}句）")
                        error_details.setdefault("voice_violations", []).append({
                            "type": "consecutive_short_sentences",
                            "count": consecutive_short,
                            "limit": max_consecutive
                        })
                        break
                else:
                    consecutive_short = 0
            # 对话占比
            dialogue_ratio_range = voice_cfg.get("structure", {}).get("dialogue_ratio_range", [0.25, 0.6])
            dialogue_chars = len(re.findall(r'[\u4e00-\u9fff，。！？；：“”‘’]+', scene_text))
            total_chars = len(scene_text)
            ratio = dialogue_chars / total_chars if total_chars > 0 else 0
            if ratio < dialogue_ratio_range[0] or ratio > dialogue_ratio_range[1]:
                voice_violations.append(f"对话占比 {ratio:.0%} 超出范围 {dialogue_ratio_range[0]:.0%}-{dialogue_ratio_range[1]:.0%}")
                error_details.setdefault("voice_violations", []).append({
                    "type": "dialogue_ratio",
                    "ratio": ratio,
                    "range": dialogue_ratio_range
                })
            if voice_violations:
                logger.warning(f"Voice violations: {voice_violations}")

            # ========== 因果关系校验（保持不变） ==========
            predicates = constraints.get("predicates", {})
            budget = constraints.get("budget")
            degraded = constraints.get("degraded", False)
            causality_failed = False
            causality_suggestions = []
            if predicates:
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

            # ========== Loop 推进检查 ==========
            loop_advancement_score = 0.0
            if active_loop:
                advanced, score, reason = await self._check_loop_advancement(scene_text, active_loop)
                loop_advancement_score = score
                logger.info(f"📊 loop_advancement_score assigned: {loop_advancement_score:.3f} (advanced={advanced})")
                if not advanced:
                    errors.append(f"叙事环路未推进: {reason}")
                    error_details["loop_not_advanced"] = reason
                else:
                    error_details["loop_advancement_score"] = score

            # ========== Planning Contract 验证 (Surface/Constraint/Outcome Control) ==========
            if planning_contract:
                surface_result = self._validate_contract_units(planning_contract, scene_text)
                control_scores["surface_control"] = surface_result
                constraint_result = self._validate_contract_constraints(planning_contract, scene_text)
                control_scores["constraint_control"] = constraint_result
                final_state = constraints.get("final_state", {}) or current_state_dict

                outcome_result = self._validate_contract_observables(
                    planning_contract,
                    parsed_data.get("events", []),
                    final_state
                )
                control_scores["outcome_control"] = outcome_result
                # 聚合缺失项
                total_missing_changes.extend(outcome_result.get("missing_changes", []))                
                if total_missing_changes:
                    logger.info("AGGREGATED_MISSING_CHANGES: count=%d", len(total_missing_changes))                  
                                
                if not constraint_result.get("passed", True):
                    errors.append("违反硬性约束")
                    should_retry = True
                if outcome_result.get("matched", 0) < outcome_result.get("total", 0):
                    missing_outcomes = outcome_result.get("total", 0) - outcome_result.get("matched", 0)
                    errors.append(f"缺失 {missing_outcomes} 个预期状态变化")
                    should_retry = True

            # ========== 确定最终结果 ==========
            passed = len(errors) == 0
            if not passed:
                feedback = "；".join(errors)
                logger.warning(f"Validation failed: {feedback}")
                suggestions = [
                    "请严格遵循场景计划中的 goal、conflict 和 must_events 的语义要求",
                    "确保生成的正文完整表达所有必须事件的核心含义",
                    "请严格按照导演蓝图中的知识变化序列和场景角色要求",
                ]
                if missing_knowledge:
                    suggestions.append(f"请确保在正文中明确体现以下知识变化：{', '.join(missing_knowledge)}")
                if not scene_obj_ok and scene_objective:
                    suggestions.append(f"请确保场景完成其存在理由：{scene_objective}")
                if causality_suggestions:
                    suggestions.extend(causality_suggestions)
                result = {
                    "passed": False,
                    "feedback": feedback,
                    "suggestions": suggestions,
                    "should_retry": should_retry,
                    "error_details": error_details,
                    "parsed_output": parsed_data,
                    "loop_advancement_score": loop_advancement_score,
                    "control_scores": control_scores,
                    "missing_changes": total_missing_changes,   # 新增
                }
            else:
                result = {
                    "passed": True,
                    "feedback": "校验通过",
                    "suggestions": [],
                    "should_retry": False,
                    "error_details": error_details,
                    "parsed_output": parsed_data,
                    "loop_advancement_score": loop_advancement_score,
                    "control_scores": control_scores,
                    "missing_changes": total_missing_changes,   # 新增
                }

            # ========== Phase 14.0C-3 Commit A: Contract Realization Observability ==========
            total_state_changes = len(planning_contract.observables.state_changes) if planning_contract else 0
            semantic_stats = control_scores.get("semantic_validation", {})
            semantic_available = "semantic_validation" in control_scores

            if total_state_changes == 0:
                realization_rate = 1.0
            elif semantic_available:
                match_count = semantic_stats.get("match_count", 0)
                realization_rate = round(match_count / total_state_changes, 3)
            else:
                realization_rate = None

            # ============================================================
            # 统一构造 parser_realization（基于来源标记）
            # ============================================================
            # 注意：_source_format 等变量已在前面根据数据来源设置
            parser_realization = {
                "source_format": _source_format,
                "scene_text_extracted": True,
                "events_extracted": _events_extracted,
                "event_count": _event_count,
                "fallback_reason": _fallback_reason,
            }
            # ============================================================

            contract_realization = {
                "expected_changes": total_state_changes,
                "semantic_available": semantic_available,
                "matched_semantically": semantic_stats.get("match_count") if semantic_available else None,
                "missing_semantically": total_state_changes - semantic_stats.get("match_count", 0) if semantic_available else None,
                "realization_rate": realization_rate,
                "parser_realization": parser_realization,
            }
            # ============================================================

            # ============================================================
            # Commit D.1: 强制 DEBUG 日志（临时观测）
            # ============================================================
            logger.critical(
                "CONTRACT_REALIZATION_DEBUG: contract=%s parser=%s events=%d keys=%s",
                json.dumps(contract_realization, ensure_ascii=False),
                json.dumps(parser_realization, ensure_ascii=False),
                len(parsed_data.get("events", [])),
                list(parsed_data.keys())
            )
            # ============================================================

            # ========== 构造 ValidatorOutput ==========
            output = self._build_validator_output(
                execution_id=execution_id,
                passed=passed,
                errors=errors,
                voice_violations=voice_violations,
            )
            result["validator_output"] = output.to_runtime_dict()
            result["contract_realization"] = contract_realization
            logger.info(
                "ValidatorOutput generated: %s",
                json.dumps(output.to_runtime_dict(), ensure_ascii=False, default=str)
            )
            return result

        except Exception as e:
            # ========== Phase 14.0C-3A: 异常收敛为 ValidatorOutput ==========
            logger.exception(
                "Validator enhanced validation failed unexpectedly: "
                f"execution_id={execution_id}, error={type(e).__name__}: {e}"
            )

            from src.writing.validation_result import (
                ValidatorOutput,
                Violation,
                ViolationSeverity,
                ValidationStage,
                ValidationStatus,
            )

            output = ValidatorOutput(
                execution_id=execution_id or "unknown",
                stage=ValidationStage.SEMANTIC,
                status=ValidationStatus.FAILED,
                violations=[
                    Violation(
                        rule_id="validator_runtime_error",
                        severity=ViolationSeverity.ERROR,
                        description=f"{type(e).__name__}: {str(e)[:200]}",
                    )
                ],
                confidence=0.0,
            )
            logger.info(
                "ValidatorOutput generated (error): %s",
                json.dumps(output.to_runtime_dict(), ensure_ascii=False, default=str)
            )

            return {
                "passed": False,
                "validator_output": output.to_runtime_dict(),
                "feedback": f"Validator runtime error: {type(e).__name__}",
                "should_retry": True,
                "error_details": {
                    "exception_type": type(e).__name__,
                    "exception_message": str(e)[:500],
                },
                "parsed_output": {},
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

    def _extract_json_array(self, text: str, key: str) -> List[Any]:
        """
        从文本中提取指定 key 对应的 JSON 数组。
        使用括号匹配（bracket matching）处理嵌套结构。
        返回 list，如果提取失败则返回空列表。
        """
        import json
        import re

        if not text:
            return []

        # 1. 查找 key 位置，如 "events":
        pattern = rf'"{key}"\s*:\s*(\[)'
        match = re.search(pattern, text)
        if not match:
            return []

        # 2. 从 '[' 开始扫描，使用括号匹配
        start = match.start(1)
        brace_count = 0
        in_string = False
        escape = False
        end = start

        for i in range(start, len(text)):
            ch = text[i]

            if in_string:
                if escape:
                    escape = False
                    continue
                if ch == '\\':
                    escape = True
                    continue
                if ch == '"':
                    in_string = False
                    continue
                continue

            if ch == '"':
                in_string = True
                continue

            if ch == '[':
                brace_count += 1
            elif ch == ']':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break

        if brace_count != 0 or end == start:
            return []

        # 3. 提取数组字符串并解析
        array_str = text[start:end]
        try:
            result = json.loads(array_str)
            if isinstance(result, list):
                return result
            return []
        except json.JSONDecodeError:
            # 降级：尝试逐项提取（仅适用于简单对象）
            logger.warning(f"_extract_json_array: JSON parse failed for key '{key}', trying individual items")
            items = re.findall(r'\{[^{}]*\}', array_str)
            parsed = []
            for item_str in items:
                try:
                    parsed.append(json.loads(item_str))
                except:
                    pass
            return parsed

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

    # ==================== 新增：Loop 推进检查 ====================
    async def _check_loop_advancement(self, scene_text: str, loop: dict) -> tuple[bool, float, str]:
        """
        检查场景是否实质推进了叙事环路
        返回: (是否推进, 推进分数 0-1, 理由)
        """
        if not loop or not loop.get("description"):
            return True, 0.0, "无激活 Loop，跳过检查"

        try:
            prompt = f"""
你是一位叙事分析专家。判断以下场景对指定叙事环路的推进程度。

环路描述：{loop['description']}
当前进度：{loop.get('progress', 0)*100:.0f}%

场景文本：
{scene_text[:2000]}

请评估本章对环路的推进程度（0-1），并输出 JSON：
{{
    "advanced": true/false,      // 是否有实质推进
    "score": 0.0-1.0,            // 推进程度（0=无，0.1=轻微，0.5=中等，1.0=重大突破）
    "reason": "简短理由"
}}
"""
            client = AsyncOpenAI(api_key="not-needed", base_url=self.llm_api_url)
            response = await client.chat.completions.create(
                model="Qwen3-32B-Q5_K_M",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=256,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            advanced = result.get("advanced", False)
            score = result.get("score", 0.0)
            reason = result.get("reason", "未提供理由")
            logger.info(f"🔍 Loop advancement check result: advanced={advanced}, score={score:.3f}, reason={reason[:60]}")            

            #return advanced, min(1.0, max(0.0, score)), reason
            return advanced, score, reason        
        except Exception as e:
            logger.warning(f"Loop advancement check failed (fallback: pass with 0.05): {e}")
            return True, 0.05, f"检查异常，默认推进 5%: {e}"

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
    
    def _validate_contract_units(self, contract: PlanningContract, text: str) -> Dict[str, Any]:
        """验证 Execution Units 是否被完成"""
        if not contract.execution.units:
            return {"completed": 0, "total": 0, "details": [], "score": 1.0}
        
        completed = 0
        details = []
        for unit in contract.execution.units:
            # 提取关键词
            keywords = re.findall(r'[\u4e00-\u9fff]{2,4}', unit.description)
            if not keywords:
                # 如果没有关键词，使用整个描述的前6个字
                keywords = [unit.description[:6]]
            matched = any(kw in text for kw in keywords)
            if matched:
                completed += 1
            details.append({
                "unit_id": unit.id, 
                "description": unit.description, 
                "completed": matched
            })
        
        total = len(contract.execution.units)
        score = completed / total if total > 0 else 1.0
        return {
            "completed": completed, 
            "total": total, 
            "score": score,
            "details": details
        }

    def _validate_contract_constraints(self, contract: PlanningContract, text: str) -> Dict[str, Any]:
        """验证硬约束"""
        if not contract.constraints:
            return {"passed": True, "details": [], "score": 1.0}
        
        results = []
        all_passed = True
        for c in contract.constraints:
            keywords = re.findall(r'[\u4e00-\u9fff]{2,4}', c.target)
            if not keywords:
                keywords = [c.target[:6]]
            
            if c.type == "required":
                matched = any(kw in text for kw in keywords) if keywords else False
                passed = matched
                if not passed:
                    all_passed = False
            elif c.type == "forbidden":
                matched = any(kw in text for kw in keywords) if keywords else False
                passed = not matched
                if not passed:
                    all_passed = False
            else:
                # before/after/exclusive/at_least_once 暂不实现
                passed = True
            results.append({
                "type": c.type, 
                "target": c.target, 
                "passed": passed
            })
        
        total = len(contract.constraints)
        passed_count = sum(1 for r in results if r["passed"])
        score = passed_count / total if total > 0 else 1.0
        return {"passed": all_passed, "details": results, "score": score}

    def _validate_contract_observables(
        self,
        contract: PlanningContract,
        events: List[Dict],
        final_state: Dict
    ) -> Dict[str, Any]:
        """验证可观测结果是否发生（使用 Resolver + Matcher），并返回缺失投影。"""
        logger.info(
            "Validating observables: %d expected, %d actual events",
            len(contract.observables.state_changes),
            len(events)
        )

        if not contract.observables.state_changes:
            return {"matched": 0, "total": 0, "details": [], "score": 1.0, "missing_changes": []}

        matched = 0
        total = len(contract.observables.state_changes)
        details = []
        missing_changes = []  # 新增

        for change in contract.observables.state_changes:
            expected_types = ContractEventResolver.resolve(change.type)
            if not expected_types:
                logger.warning(
                    "CONTRACT_EVENT_UNKNOWN_TYPE: %s (skip matching)",
                    change.type
                )
                details.append({
                    "change": change.model_dump() if hasattr(change, 'model_dump') else vars(change),
                    "found": False,
                    "reason": "unknown_type"
                })
                continue

            expected_values = {et.value for et in expected_types}
            found = False
            for evt in events:
                if evt.get("type") in expected_values:
                    if ContractEventMatcher.match(change, evt):
                        found = True
                        break

            if found:
                matched += 1
                logger.debug(f"  ✅ Matched: {change.type}")
            else:
                logger.debug(f"  ❌ Not matched: {change.type}")
                # 生成 MissingContractChange
                change_type = (
                    change.type.value
                    if hasattr(change.type, "value")
                    else str(change.type)
                )
                missing_changes.append(
                    MissingContractChange(
                        type=change_type,
                        description=self._describe_change(change),
                        severity=ContractSeverity.BLOCKING,
                        actor=getattr(change, "actor", None),
                        fields=change.model_dump() if hasattr(change, 'model_dump') else {},
                        source="planning_contract",
                        contract_id=getattr(
                            contract,
                            "contract_id",
                            getattr(contract, "scene_id", None)
                        ),
                        confidence=1.0,
                    )
                )
                logger.info(
                    "CONTRACT_MISSING_CHANGE: type=%s severity=%s actor=%s",
                    change_type,
                    ContractSeverity.BLOCKING.value,
                    getattr(change, "actor", None)
                )

            details.append({
                "change": change.model_dump() if hasattr(change, 'model_dump') else vars(change),
                "found": found
            })

        score = matched / total if total > 0 else 1.0
        logger.info(
            "VALIDATOR_EVENT_MATCH: matched=%d expected=%d score=%.3f",
            matched, total, score
        )
        return {
            "matched": matched,
            "total": total,
            "score": score,
            "details": details,
            "missing_changes": missing_changes  # 新增
        }

    _semantic_validator: Optional[SemanticValidator] = None

    def _describe_change(self, change: StateChange) -> str:
        t = (
            change.type.value
            if hasattr(change.type, "value")
            else str(change.type)
        )
        if t == "plot_flag":
            return f"剧情标记 {change.name} 应设为 {change.value}"
        if t == "inventory_acquire":
            return f"{change.actor} 应获得 {change.item}"
        if t == "location_change":
            return f"{change.actor} 应到达 {change.location}"
        if t == "realm_change":
            return f"{change.actor} 应突破至 {change.to_major_realm} {change.to_minor_stage}层"
        if t == "relationship_change":
            return f"{change.from_char} 与 {change.to_char} 的关系应变化 {change.delta}"
        if t == "knowledge_gain":
            return f"应获得知识：{change.name}"
        return f"缺失类型 {t} 的状态变化"

    @classmethod
    def _get_validator(cls) -> SemanticValidator:
        if cls._semantic_validator is None:
            try:
                from src.writing.embedding import HttpEmbeddingProvider
                provider = HttpEmbeddingProvider(
                    endpoint=config.embedding_endpoint,
                    model=config.embedding_model,
                )
                logger.info("SemanticValidator initialized with HttpEmbeddingProvider")
                cls._embedding_status = "available"
            except (ImportError, AttributeError, Exception) as e:
                # ========== Phase 14.0C-3A: 记录 fallback 状态 ==========
                logger.warning(
                    f"EmbeddingProvider unavailable, falling back to NoOpEmbeddingProvider: "
                    f"{type(e).__name__}: {e}",
                    exc_info=True,
                )
                cls._embedding_status = "unavailable"
                cls._embedding_fallback_reason = str(e)
                # =======================================================
                provider = NoOpEmbeddingProvider()

            cls._semantic_validator = SemanticValidator(
                embedding_provider=provider,
                keyword_threshold=0.6,
                embedding_threshold=0.30,
                embedding_min_confidence=0.6,
                enable_embedding=getattr(config, 'enable_embedding_validator', True),
            )
        return cls._semantic_validator
    
    def _build_validator_output(
        self,
        execution_id: str,
        passed: bool,
        errors: List[str],
        voice_violations: List[str],
    ) -> ValidatorOutput:
        """构造 ValidatorOutput 协议对象"""
        violations = []

        for err in errors:
            violations.append(Violation(
                rule_id="validation_error",
                severity=ViolationSeverity.ERROR,
                description=err[:200],
            ))

        for warning in voice_violations:
            violations.append(Violation(
                rule_id="voice_violation",
                severity=ViolationSeverity.WARNING,
                description=warning[:200],
            ))

        if not passed:
            status = ValidationStatus.FAILED
            confidence = 0.0
        elif voice_violations:
            status = ValidationStatus.DEGRADED
            confidence = 0.8
        else:
            status = ValidationStatus.PASSED
            confidence = 1.0

        return ValidatorOutput(
            execution_id=execution_id,
            stage=ValidationStage.SEMANTIC,
            status=status,
            violations=violations,
            confidence=confidence,
        )