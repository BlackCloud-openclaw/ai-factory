# src/writing/controlled_writer.py
"""
Controlled Writer - 产品化增量执行服务

Phase 13.2.3C: 集成 QualityGate 实现控制闭环
Phase 15.7-A: 注入 Rewriter 依赖（暂不执行）
"""

import re
import json
import time
import asyncio
import pathlib                                    # ← 新增
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field, ValidationError
from openai import AsyncOpenAI
import httpx
from enum import Enum

from src.model_router import get_router
from src.execution.llm_router_pool import get_llm_router_pool
from src.config import config

from src.writing.planning_contract import PlanningContract, ExecutionUnit
from src.writing.contracts import WritingContract, WritingConstraints, WritingGoal
from src.writing.scene_execution_context import SceneExecutionContext
from src.writing.narrative_intent import NarrativeIntent
from src.config.settings import settings
from src.writing.runtime import RuntimeServices

# Phase 13.2.3C 导入
from .validation import SemanticValidator, ValidationResult
from .quality_gate import QualityGate, QualityGateResult

# ========== Phase 15.7-A: Rewriter 导入 ==========
from src.writing.shadow.runner import Rewriter
from src.common.logging import setup_logging


logger = setup_logging("writing.controlled_writer")

class WriterOutput(BaseModel):
    """LLM 输出结构验证"""
    scene_text: str = Field(..., min_length=50, description="场景正文，至少50字")
    events: List[Dict] = Field(default_factory=list, description="状态变化事件列表")
    foreshadowing: List[str] = Field(default_factory=list, description="伏笔列表")


# ============================================================================
# Phase 15.8 Commit 1: Rewrite Selection Contract
# ============================================================================

class RewriteSelectionReason(str, Enum):
    """
    Selection 原因常量。

    Commit 1 只可能出现前 5 种；SELECTED 属于 Commit 3 启用路径。
    """
    NO_REWRITER_INJECTED       = "no_rewriter_injected"
    TEXT_TOO_SHORT             = "text_too_short"
    MISSING_EXECUTION_CONTRACT = "missing_execution_contract"
    REWRITE_UNAVAILABLE        = "rewrite_unavailable"
    STRUCTURAL_UNSAFE          = "structural_unsafe"   # Commit 1 默认路径
    SELECTED                   = "selected"            # Commit 3 才启用


@dataclass
class RewriteSelectionResult:
    """
    Phase 15.8 Commit 1: Rewrite Selection Contract

    不变量（Commit 1）：
        selected_source == "original"
        final_text == original_text
        structural_safe 恒为 False
    """
    selected_source: str          # "original" | "rewritten"
    selection_reason: str         # RewriteSelectionReason.value
    rewrite_available: bool       # Rewriter 是否成功产出可用文本
    structural_safe: bool         # Commit 1 恒 False
    rewrite_attempted: bool
    rewrite_failure_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_source": self.selected_source,
            "selection_reason": self.selection_reason,
            "rewrite_available": self.rewrite_available,
            "structural_safe": self.structural_safe,
            "rewrite_attempted": self.rewrite_attempted,
            "rewrite_failure_reason": self.rewrite_failure_reason,
        }


@dataclass
class ControlledWriteResult:
    """
    ControlledWriter 执行结果。

    Phase 15.7-A 扩展:
    - original_text: 原始 draft
    - rewritten_text: Rewriter 产出（15.7-A 始终为 None）
    - rewrite_attempted: 是否尝试 Rewrite（15.7-A 始终为 False）
    - rewrite_failure_reason: 失败原因（15.7-A 始终为 None）
    """
    # 原有字段
    text: str
    events: List[Dict]
    segments_used: int
    segments_succeeded: int
    fallback_used: bool
    execution_time: float

    # ========== Phase 15.7-A: 预留双轨字段 ==========
    # ========== Phase 15.7-A: 预留双轨字段 ==========
    original_text: str = ""
    rewritten_text: Optional[str] = None
    rewrite_attempted: bool = False
    rewrite_failure_reason: Optional[str] = None
    # =============================================

    # ========== Phase 15.8 Commit 1: Selection Contract ==========
    selection: Optional[RewriteSelectionResult] = None
    # =============================================================

    def __post_init__(self):
        # 确保 text == original_text（15.7-A 不变量）
        if not self.original_text:
            object.__setattr__(self, 'original_text', self.text)


class ControlledWriter:
    """
    受控写入器。

    支持通过 runtime_services 注入 Runtime 服务。
    Phase 13.2.3C: 注入 SemanticValidator 和 QualityGate 实现控制闭环。
    Phase 15.7-A: 注入 Rewriter 依赖（暂不执行）。
    Phase 15.8-fix: 加载 GBNF grammar 强制 JSON 输出。
    """

    # ========== Phase 15.8-fix: Grammar 缓存（类级） ==========
    _grammar: Optional[str] = None

    @classmethod
    def _get_grammar(cls) -> Optional[str]:
        """
        加载 GBNF grammar 文件，用于强制 LLM 输出合法 JSON。

        与 WritingAgent._get_grammar 行为一致：
        - 成功 → 返回 grammar 字符串
        - 文件不存在或读取失败 → 返回 None（不影响正常运行）
        - 空字符串表示"已尝试但失败"，避免重复 IO
        """
        if cls._grammar is not None:
            return cls._grammar if cls._grammar else None

        grammar_path = (
            pathlib.Path(__file__).parent.parent.parent
            / "grammars" / "json_writer.gbnf"
        )
        if grammar_path.exists():
            try:
                cls._grammar = grammar_path.read_text(encoding="utf-8")
                logger.info(
                    "ControlledWriter: loaded grammar from %s", grammar_path
                )
                return cls._grammar
            except Exception as e:
                logger.error(
                    "ControlledWriter: failed to load grammar: %s", e
                )
                cls._grammar = ""
                return None
        else:
            logger.warning(
                "ControlledWriter: grammar not found at %s, "
                "JSON enforcement will rely on model only",
                grammar_path,
            )
            cls._grammar = ""
            return None
    # ============================================================

    def __init__(
        self,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        max_retries_per_segment: int = 2,
        enable_fallback: bool = True,
        runtime_services: Optional[RuntimeServices] = None,
        semantic_validator: Optional[SemanticValidator] = None,
        quality_gate: Optional[QualityGate] = None,
        # ========== Phase 15.7-A: 接收 Rewriter 依赖 ==========
        rewriter: Optional[Rewriter] = None,
        # =====================================================
    ):
        self.api_base = api_base or settings.llm_api_url
        self.model = model or getattr(settings, 'llm_writing_model', 'Qwen3-32B-Q5_K_M')
        self.max_retries_per_segment = max_retries_per_segment
        self.enable_fallback = enable_fallback
        self._runtime_services = runtime_services

        # Phase 13.2.3C: 注入 Validator 和 QualityGate
        self._semantic_validator = semantic_validator or SemanticValidator()
        self.quality_gate = quality_gate or QualityGate()

        # ========== Phase 15.7-A: 存储 Rewriter ==========
        self._rewriter = rewriter
        # =================================================

        # ========== Phase 15.7-B1: 初始化状态变量 ==========
        self._parse_failure_logged = False
        # =================================================

    # ========================================================================
    # 原有方法（保持不变）
    # ========================================================================

    def _determine_segments(self, units: List[ExecutionUnit]) -> int:
        total = len(units)
        if total <= 4:
            return 1
        elif total <= 8:
            return 2
        else:
            return 3

    def _split_units(self, units: List[ExecutionUnit], segments: int) -> List[List[ExecutionUnit]]:
        if segments == 1:
            return [units]
        total = len(units)
        base = total // segments
        remainder = total % segments
        result = []
        idx = 0
        for i in range(segments):
            count = base + (1 if i < remainder else 0)
            if count == 0:
                count = 1
            result.append(units[idx:idx + count])
            idx += count
        if idx < total:
            result[-1].extend(units[idx:])
        return result

    def _build_segment_prompt(
        self,
        writing_contract: WritingContract,
        segment_units: List[ExecutionUnit],
        segment_idx: int,
        total_segments: int,
        previous_text: str,
        previous_events: List[Dict],
        current_state: Dict,
        is_retry: bool = False,
        is_fallback: bool = False,
        error_hint: str = "",
    ) -> str:
        lines = []
        
        # ========== 1. NarrativeIntent 指令 ==========
        narrative_intent = writing_contract.narrative_intent
        if narrative_intent is not None and hasattr(narrative_intent, 'scene_role'):
            from src.writing.narrative_intent import NarrativeContext
            try:
                context = NarrativeContext.from_intent(narrative_intent)
                lines.append(context.to_prompt_instructions())
                lines.append("")
            except Exception:
                pass
        
        # ========== 2. 场景目标 ==========
        if hasattr(writing_contract, 'writing_goal') and writing_contract.writing_goal:
            goal = writing_contract.writing_goal
            lines.append("【场景目标】")
            if goal.goal:
                lines.append(f"目标：{goal.goal}")
            if goal.conflict:
                lines.append(f"冲突：{goal.conflict}")
            if goal.expected_outcome:
                lines.append(f"预期结果：{goal.expected_outcome}")
            lines.append("")

        # ========== 2.5 Phase 15.8 Commit 2: 上一场景结尾（仅第 0 段） ==========
        if segment_idx == 0 and not previous_text:
            prev_tail = getattr(writing_contract, 'previous_scene_tail', None)
            if prev_tail:
                lines.append("【上一场景结尾（供自然衔接参考）】")
                lines.append(prev_tail[-300:])
                lines.append("请自然衔接上一场景结尾的剧情与动作，避免重复已发生的情节。")
                lines.append("")
        # ========================================================================

        # ========== 3. 分段说明 ==========
        if is_fallback:
            lines.append("⚠️ 降级模式：一次性生成完整场景，约 800-1200 字。")
            lines.append("")
        else:
            lines.append(f"请写一段场景正文（约 400-600 字）。这是第 {segment_idx + 1}/{total_segments} 段。")
            lines.append("")
        
        # ========== 4. 上一段结尾（上下文衔接） ==========
        if previous_text and len(previous_text) > 50:
            lines.append("【上一段结尾】")
            lines.append(previous_text[-300:])
            lines.append("请自然衔接上一段结尾。")
            lines.append("")
        
        # ========== 5. 已完成的事件摘要 ==========
        if previous_events:
            lines.append("【已完成的事件摘要】")
            for evt in previous_events[-5:]:
                evt_type = evt.get("type", "unknown")
                if evt_type == "plot_flag_set":
                    lines.append(f"  - 触发标记：{evt.get('flag')} = {evt.get('value', True)}")
                elif evt_type == "item_acquire":
                    lines.append(f"  - {evt.get('actor')} 获得 {evt.get('item')}")
                elif evt_type == "location_enter":
                    lines.append(f"  - {evt.get('actor')} 进入 {evt.get('location')}")
                elif evt_type == "realm_upgrade":
                    lines.append(f"  - {evt.get('actor')} 突破到 {evt.get('to_major_realm')}{evt.get('to_minor_stage')}层")
                elif evt_type == "relationship_change":
                    lines.append(f"  - {evt.get('from_char')} 与 {evt.get('to_char')} 关系变化 {evt.get('delta')}")
            lines.append("")
        
        # ========== 6. 当前世界状态摘要 ==========
        if current_state:
            lines.append("【当前世界状态】")
            chars = current_state.get("characters", {})
            for name, info in list(chars.items())[:3]:
                hp = info.get("hp", "?")
                realm = info.get("realm", "?")
                level = info.get("level", 1)
                location = info.get("location", "未知")
                lines.append(f"  {name}: 境界={realm}{level}层, HP={hp}, 位置={location}")
            lines.append("")
        
        # ========== 7. 本段必须完成的执行单元 ==========
        if segment_units:
            lines.append("【本段必须完成的执行单元】")
            for unit in segment_units:
                lines.append(f"- {unit.description}")
            lines.append("")
        
        # ========== 8. 硬性约束 ==========
        if hasattr(writing_contract, 'constraints') and writing_contract.constraints:
            constraints = writing_contract.constraints
            if constraints.must_events:
                lines.append("【必须发生的事件】")
                for evt in constraints.must_events:
                    lines.append(f"  ✅ {evt}")
                lines.append("")
            if constraints.forbidden_events:
                lines.append("【禁止发生的事件】")
                for evt in constraints.forbidden_events:
                    lines.append(f"  ❌ {evt}")
                lines.append("")
        
        # ========== 9. 重试反馈（Phase 13.2.3C） ==========
        if error_hint:
            lines.append(f"⚠️ 上一轮验证反馈：{error_hint}")
            lines.append("请根据以上反馈修正生成内容。")
            lines.append("")
        
        # ========== 10. 输出格式要求 ==========
        lines.append("【输出格式】")
        lines.append('{"scene_text": "场景正文（纯文本）", "events": [{"type": "...", ...}], "foreshadowing": ["伏笔1", "伏笔2"]}')
        lines.append("只输出 JSON，不要有任何额外文本。")
        
        return "\n".join(lines)

    def _verify_segment(self, text: str, units: List[ExecutionUnit]) -> bool:
        if not units:
            return True
        if len(text.strip()) < 200:
            return False
        for unit in units:
            keywords = re.findall(r'[\u4e00-\u9fff]{2,4}', unit.description)
            if not keywords:
                keywords = [unit.description[:6]]
            if not any(kw in text for kw in keywords):
                return False
        return True

    def _parse_and_validate(self, text: str) -> Optional[WriterOutput]:
        if not text:
            return None
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
            if not data.get("scene_text") or len(data["scene_text"].strip()) < 50:
                logger.warning("scene_text 缺失或过短")
                return None
            return WriterOutput(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"JSON 解析或验证失败: {e}")
            return None

    def _apply_events(self, events: List[Dict], state: Dict) -> Dict:
        state = state.copy()
        if not state:
            state = {"characters": {"林逸": {"hp": 100, "realm": "炼气", "level": 1, "inventory": []}}, "global_flags": {}}

        for evt in events:
            evt_type = evt.get("type", "")
            if evt_type == "plot_flag_set":
                flag = evt.get("flag")
                if flag:
                    state.setdefault("global_flags", {})[flag] = evt.get("value", True)
            elif evt_type == "item_acquire":
                actor = evt.get("actor")
                item = evt.get("item")
                if actor and item:
                    if actor not in state["characters"]:
                        state["characters"][actor] = {"inventory": []}
                    if "inventory" not in state["characters"][actor]:
                        state["characters"][actor]["inventory"] = []
                    state["characters"][actor]["inventory"].append(item)
            # ... 其他事件类型 ...
        return state

    async def _call_llm(self, prompt: str, max_tokens: int = 2048) -> tuple[str, dict]:
        logger.critical("[15.7-B1] === _call_llm ENTERED ===")
        logger.critical("[15.7-B1] _call_llm: prompt_len=%d, max_tokens=%d", len(prompt), max_tokens)
        
        # 检查配置
        logger.critical("[15.7-B1] _call_llm: self.api_base=%s, self.model=%s", self.api_base, self.model)
        
        router = get_router()
        primary_model = router.get_model_for_task("writing")
        fallback_model = "Qwen3-32B-Q5_K_M"
        pool = get_llm_router_pool()
        
        logger.critical("[15.7-B1] _call_llm: primary_model=%s, fallback_model=%s", primary_model, fallback_model)
        
        # 定义实际调用函数
        async def _do_call(model_name: str, **kwargs) -> tuple[str, dict]:
            logger.critical("[15.7-B1] _do_call: model=%s", model_name)
            base_url = kwargs.get('base_url') or self.api_base
            logger.critical("[15.7-B1] _do_call: base_url=%s", base_url)
            
            transport = httpx.AsyncHTTPTransport(proxy=None)
            async with httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(600.0, connect=30.0)) as client:
                openai_client = AsyncOpenAI(
                    api_key="not-needed",
                    base_url=base_url,
                    http_client=client,
                )

                # ========== Phase 15.8-fix: 附加 GBNF grammar ==========
                extra_body = {}
                grammar_str = self._get_grammar()
                if grammar_str:
                    extra_body["grammar"] = grammar_str
                    logger.critical(
                        "[15.7-B1] _do_call: grammar attached (len=%d)",
                        len(grammar_str),
                    )
                else:
                    logger.critical(
                        "[15.7-B1] _do_call: grammar NOT attached (fallback to response_format only)"
                    )
                # ========================================================

                logger.critical("[15.7-B1] _do_call: sending request to OpenAI...")
                response = await openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    extra_body=extra_body if extra_body else None,   # ← 新增
                )
                content = response.choices[0].message.content or ""
                usage = response.usage.model_dump() if response.usage else {"total_tokens": 0}
                logger.critical("[15.7-B1] _do_call: response_len=%d, usage=%s", len(content), usage)
                return content, usage
        
        logger.critical("[15.7-B1] _call_llm: calling pool.call with primary_model=%s", primary_model)
        try:
            result = await pool.call(primary_model, _do_call, timeout=getattr(config, 'llm_timeout_writing', 600), agent="writer")
            logger.critical("[15.7-B1] _call_llm: pool.call returned, result_len=%d", len(result))
            return result
        except Exception as e:
            logger.critical("[15.7-B1] _call_llm: primary_model failed: %s", e, exc_info=True)
            try:
                logger.critical("[15.7-B1] _call_llm: trying fallback_model=%s", fallback_model)
                result = await pool.call(fallback_model, _do_call, timeout=getattr(config, 'llm_timeout_writing', 600), agent="writer")
                logger.critical("[15.7-B1] _call_llm: fallback succeeded, result_len=%d", len(result))
                return result
            except Exception as e2:
                logger.critical("[15.7-B1] _call_llm: fallback also failed: %s", e2, exc_info=True)
                raise
    # ========================================================================
    # Phase 13.2.3C: 核心 segment 执行 (集成 QualityGate)
    # ========================================================================

    async def _execute_segment(
        self,
        contract: WritingContract,
        units: List[ExecutionUnit],
        idx: int,
        total: int,
        previous_text: str,
        previous_events: List[Dict],
        current_state: Dict,
    ) -> Tuple[str, List[Dict], bool]:
        """
        执行单个 Segment，集成 QualityGate 实现控制闭环。

        关键修复 (Phase 13.2.3C v1.1):
            - error_hint 在循环外初始化，跨 attempt 保留
            - feedback 注入下一轮 prompt
            - 安全返回 fallback
        """
        logger.critical("[15.7-B1] === _execute_segment ENTERED ===")
        text = ""
        events = []
        error_hint = ""  # ✅ 在循环外初始化，跨 attempt 保留

        for attempt in range(self.max_retries_per_segment + 1):
            is_retry = attempt > 0

            prompt = self._build_segment_prompt(
                writing_contract=contract,
                segment_units=units,
                segment_idx=idx,
                total_segments=total,
                previous_text=previous_text,
                previous_events=previous_events,
                current_state=current_state,
                is_retry=is_retry,
                error_hint=error_hint,  # ✅ 传入累积的 feedback
            )

            try:
                max_tokens = 4096 if attempt > 1 else 2048
                response_content, usage = await self._call_llm(prompt, max_tokens=max_tokens)
                # ========== D.3 观测点 1：LLM 原始响应 ==========
                logger.critical(
                    "WRITER_LLM_RAW: len=%d has_events_key=%s preview=%s",
                    len(response_content),
                    '"events"' in response_content,
                    response_content[:500]
                )
                # =============================================                
                validated = self._parse_and_validate(response_content)
                # ========== D.3 观测点 2：解析后 Artifact ==========
                if validated:
                    logger.critical(
                        "WRITER_SEGMENT_PARSED: scene_text_len=%d events_len=%d events_type=%s",
                        len(validated.scene_text),
                        len(validated.events),
                        type(validated.events).__name__
                    )
                    # ========== PHASE 15.0 AUDIT ==========
                    import re
                    logger.critical(
                        "[PHASE15] controlled_writer_segment parsed contains_linyi=%s abcd=%s text_len=%s",
                        "林逸" in validated.scene_text,
                        re.findall(r'\b[A-D]\b', validated.scene_text),
                        len(validated.scene_text)
                    )
                    # ====================================                    
                else:
                    # 解析失败：完整响应落盘（仅第一次，防 IO 风暴）
                    if not self._parse_failure_logged:
                        self._parse_failure_logged = True
                        from pathlib import Path
                        import datetime
                        debug_dir = Path("logs/debug")
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        fname = debug_dir / f"writer_parse_failed_{timestamp}.json"
                        fname.write_text(response_content, encoding="utf-8")
                        logger.critical(
                            "WRITER_PARSE_FAILED_LENGTH=%d saved_to=%s",
                            len(response_content),
                            fname
                        )
                # =================================================

                if validated:
                    text = validated.scene_text
                    events = validated.events

                    # P0-13: 段级只做结构验证（长度 + 单元匹配），
                    # 不做场景级契约匹配（契约需要所有段拼接后才满足）
                    validation_result = await self._validate_segment(text, contract, units)

                    # QualityGate 决策
                    gate_result = self.quality_gate.evaluate(
                        validation_result,
                        retry_count=attempt,
                        max_retries=self.max_retries_per_segment
                    )

                    if gate_result.decision in ("pass", "force_pass"):
                        logger.info(f"  ✅ 段 {idx+1} {gate_result.decision} (尝试 {attempt+1}, 分数 {gate_result.score:.2f})")
                        return text, events, True
                    else:
                        # ✅ 累积 feedback，供下一轮使用
                        error_hint = gate_result.feedback
                        logger.warning(f"  ⚠️ 段 {idx+1} {gate_result.decision} (尝试 {attempt+1}, 分数 {gate_result.score:.2f})")
                        continue
                else:
                    error_hint = "格式错误，请输出有效的 JSON。"
                    logger.warning(f"  ⚠️ 段 {idx+1} 解析失败 (尝试 {attempt+1})")

            except Exception as e:
                error_hint = f"生成异常: {e}"
                logger.warning(f"  ⚠️ 段 {idx+1} 异常 (尝试 {attempt+1}): {e}")

        # 重试耗尽，尝试降级
        if self.enable_fallback:
            logger.warning(f"  🔄 段 {idx+1} 降级到单次生成")
            # 修复：使用 execution_contract 而非 execution
            exec_units = []
            if hasattr(contract, 'execution_contract') and contract.execution_contract:
                exec_units = contract.execution_contract.execution.units
            fallback_prompt = self._build_segment_prompt(
                writing_contract=contract,
                segment_units=exec_units,
                segment_idx=0,
                total_segments=1,
                previous_text="",
                previous_events=[],
                current_state={},
                is_retry=False,
                is_fallback=True,
                error_hint="降级模式：请一次性生成完整场景。",
            )
            try:
                fb_response, _ = await self._call_llm(fallback_prompt, max_tokens=4096)
                validated = self._parse_and_validate(fb_response)
                if validated and len(validated.scene_text.strip()) > 300:
                    logger.info(f"  ✅ 降级成功 (字数 {len(validated.scene_text)})")
                    return validated.scene_text, validated.events, True
            except Exception as e:
                logger.error(f"  ❌ 降级失败: {e}")

        return "", [], False

    # ========================================================================
    # Phase 13.2.3C: 验证辅助方法
    # ========================================================================

    async def _validate_segment(
        self,
        text: str,
        contract: WritingContract,
        units: List[ExecutionUnit],
    ) -> ValidationResult:
        """
        P0-13: 段级结构验证。

        不对段级中间态做场景级契约验证——因为 contract 是场景级的
        （所有段拼接完成才满足），段级验证必然 matched=0，
        会导致 QualityGate 恒为 0.00 → 每段 3 次 retry。

        段级只做：
        - 文本长度合理（>= 200 字）
        - 分配的执行单元关键词出现（复用 _verify_segment 逻辑）

        场景级契约验证交给 ValidatorAgent（validate_node 中）。
        """
        if not text or len(text.strip()) < 200:
            return ValidationResult(
                passed=False,
                missing=["segment_too_short"],
                matched=[],
                blocking_missing=["segment_too_short"],
                overall_confidence=0.0,
                weight_applied=0.0,
            )

        if not self._verify_segment(text, units):
            return ValidationResult(
                passed=False,
                missing=["segment_units_not_covered"],
                matched=[],
                blocking_missing=["segment_units_not_covered"],
                overall_confidence=0.3,
                weight_applied=0.3,
            )

        return ValidationResult(
            passed=True,
            missing=[],
            matched=[],
            blocking_missing=[],
            overall_confidence=1.0,
            weight_applied=1.0,
        )

    # ========================================================================
    # Phase 15.8 Commit 1: Rewrite Selection
    # ========================================================================
    async def _select_rewrite(
        self,
        original_text: str,
        execution_contract: Optional[PlanningContract],
    ) -> Tuple[RewriteSelectionResult, Optional[str]]:
        """
        Phase 15.8 Commit 1: 严格按 6 级决策返回 selection。

        决策顺序：
            1. rewriter 未注入          → no_rewriter_injected
            2. text < 50 chars          → text_too_short
            3. execution_contract 空    → missing_execution_contract
            4. rewrite 异常/返回空      → rewrite_unavailable
            5. structural_safe = False  → structural_unsafe   ← Commit 1 硬编码
            6. structural_safe = True   → selected            ← Commit 3 才启用

        返回: (RewriteSelectionResult, rewritten_text 或 None)
        rewritten_text 始终保留用于双轨观察，即使 selected_source = "original"。
        """

        def _mk_original(
            reason: RewriteSelectionReason,
            available: bool,
            attempted: bool,
            failure: Optional[str],
            rewritten: Optional[str] = None,
        ) -> Tuple[RewriteSelectionResult, Optional[str]]:
            return (
                RewriteSelectionResult(
                    selected_source="original",
                    selection_reason=reason.value,
                    rewrite_available=available,
                    structural_safe=False,   # Commit 1 恒 False
                    rewrite_attempted=attempted,
                    rewrite_failure_reason=failure,
                ),
                rewritten,
            )

        # 1. rewriter 未注入
        if self._rewriter is None:
            return _mk_original(RewriteSelectionReason.NO_REWRITER_INJECTED, False, False, None)

        # 2. 文本过短
        if not original_text or len(original_text.strip()) < 50:
            return _mk_original(RewriteSelectionReason.TEXT_TOO_SHORT, False, False, None)

        # 3. execution_contract 为空
        if execution_contract is None:
            return _mk_original(
                RewriteSelectionReason.MISSING_EXECUTION_CONTRACT,
                False, False, "Missing execution_contract",
            )

        # 4. 尝试 rewrite，最多 3 次（1 原始 + 2 重试）
        #    Phase 15.8-fix5: 超时/空返回 → 重试；其他异常 → 不重试
        MAX_REWRITE_ATTEMPTS = 3
        RETRY_BACKOFF_SEC = 1.0

        rewritten_text: Optional[str] = None
        failure_reason: Optional[str] = None

        for attempt in range(MAX_REWRITE_ATTEMPTS):
            should_retry = False
            try:
                candidate = await self._rewriter.rewrite(original_text, execution_contract)
                if candidate and len(candidate.strip()) >= 50:
                    rewritten_text = candidate
                    if attempt > 0:
                        logger.info(
                            "[15.8-C1] Rewriter succeeded on attempt %d/%d",
                            attempt + 1, MAX_REWRITE_ATTEMPTS,
                        )
                    break
                else:
                    failure_reason = "Rewriter returned empty or too short"
                    should_retry = True
            except (TimeoutError, asyncio.TimeoutError) as e:
                failure_reason = f"timeout: {type(e).__name__}: {e}"
                should_retry = True
                logger.warning(
                    "[15.8-C1] Rewriter timeout (attempt %d/%d): %s",
                    attempt + 1, MAX_REWRITE_ATTEMPTS, e,
                )
            except Exception as e:
                # 非超时异常：不重试，直接 fallback
                failure_reason = f"{type(e).__name__}: {e}"
                logger.error(
                    "[15.8-C1] Rewriter exception (attempt %d/%d, no retry): %s",
                    attempt + 1, MAX_REWRITE_ATTEMPTS, e, exc_info=True,
                )
                break

            if should_retry and attempt < MAX_REWRITE_ATTEMPTS - 1:
                logger.warning(
                    "[15.8-C1] Rewriter attempt %d/%d failed (%s), retrying in %.1fs...",
                    attempt + 1, MAX_REWRITE_ATTEMPTS, failure_reason, RETRY_BACKOFF_SEC,
                )
                await asyncio.sleep(RETRY_BACKOFF_SEC)

        if rewritten_text is None:
            logger.warning(
                "[15.8-C1] Rewriter failed after %d attempts, fallback to original: %s",
                MAX_REWRITE_ATTEMPTS, failure_reason,
            )
            return _mk_original(
                RewriteSelectionReason.REWRITE_UNAVAILABLE,
                False, True, failure_reason,
            )

        # ========== Phase 15.8 Commit 3B: Structural Lock FLIP ==========
        # 唯一 flip 点：从硬编码 False → 真实 StructuralLock 判定
        structural_safe = False  # fallback if check fails
        try:
            from src.writing.structural_lock import StructuralLock
            _lock = await StructuralLock().check_async(
                original_text=original_text,
                rewritten_text=rewritten_text,
                contract=execution_contract,
            )
            structural_safe = _lock.structural_safe
            logger.info(
                "[15.8-C3B] FLIP: structural_safe=%s checks=%s summary=%s",
                structural_safe,
                {c.name: c.passed for c in _lock.checks},
                _lock.failure_summary or "-",
            )
        except Exception as _e:
            logger.error("[15.8-C3B] StructuralLock failed (fallback to False): %s", _e, exc_info=True)
            structural_safe = False
        # ==================================================================
        if not structural_safe:
            return _mk_original(
                RewriteSelectionReason.STRUCTURAL_UNSAFE,
                True, True, None, rewritten_text,
            )

        # 6. Commit 3 路径（当前不可达）
        return (
            RewriteSelectionResult(
                selected_source="rewritten",
                selection_reason=RewriteSelectionReason.SELECTED.value,
                rewrite_available=True,
                structural_safe=True,
                rewrite_attempted=True,
                rewrite_failure_reason=None,
            ),
            rewritten_text,
        )

    # ========================================================================
    # Phase 15.7-A: execute() 行为完全不变，不调用 Rewriter
    # ========================================================================
    async def execute(self, contract: WritingContract) -> ControlledWriteResult:
        """
        执行受控写入（入口方法）。

        Phase 15.7-B1: 实际调用 Rewriter，但仅做观察，不改变生产文本。
        """
        start = time.time()

        # ========== Phase 15.7-B1: 诊断日志 ==========
        logger.critical("[15.7-B1] === ControlledWriter.execute ENTERED ===")
        logger.critical(
            "[15.7-B1] contract type: %s, has execution_contract: %s, has narrative_intent: %s",
            type(contract).__name__,
            hasattr(contract, 'execution_contract'),
            hasattr(contract, 'narrative_intent'),
        )
        if hasattr(contract, 'execution_contract'):
            logger.critical(
                "[15.7-B1] execution_contract is None? %s",
                contract.execution_contract is None
            )
        if hasattr(contract, 'narrative_intent'):
            logger.critical(
                "[15.7-B1] narrative_intent is None? %s",
                contract.narrative_intent is None
            )
        # =============================================

        # 获取执行单元（标准路径，无 fallback）
        units = []
        if hasattr(contract, 'execution_contract') and contract.execution_contract:
            if hasattr(contract.execution_contract, 'execution'):
                units = contract.execution_contract.execution.units
            elif isinstance(contract.execution_contract, dict):
                units = contract.execution_contract.get("execution", {}).get("units", [])

        logger.critical("[15.7-B1] units count: %d", len(units))

        if not units:
            logger.error("[15.7-B1] No execution units found, returning empty (Contract incomplete)")
            # ❌ 直接返回空结果，不进行任何 fallback，保持 B1 硬边界
            return ControlledWriteResult(
                text="",
                events=[],
                segments_used=0,
                segments_succeeded=0,
                fallback_used=False,
                execution_time=time.time() - start,
                original_text="",
                rewritten_text=None,
                rewrite_attempted=False,
                rewrite_failure_reason="Missing execution_contract or units",
            )

        # ========== PHASE 15.0 AUDIT ==========
        scene_id = getattr(contract, 'scene_context', None)
        scene_id_str = scene_id.scene_id if scene_id else "unknown"
        chars = getattr(contract, 'scene_context', None)
        chars_list = chars.characters if chars else []
        logger.critical(
            "[PHASE15] controlled_writer_execute scene_id=%s characters=%s units_count=%s has_intent=%s",
            scene_id_str,
            chars_list,
            len(units),
            contract.narrative_intent is not None
        )
        # =====================================

        segments = self._determine_segments(units)
        segment_units = self._split_units(units, segments)

        logger.info(f"📝 ControlledWriter: {len(units)} 单元 → {segments} 段")

        text = ""
        events = []
        state = {}
        succeeded = 0
        fallback = False

        for idx, seg_units in enumerate(segment_units):
            seg_text, seg_events, ok = await self._execute_segment(
                contract=contract,
                units=seg_units,
                idx=idx,
                total=segments,
                previous_text=text,
                previous_events=events,
                current_state=state,
            )
            if ok:
                text += seg_text + "\n\n"
                events.extend(seg_events)
                state = self._apply_events(seg_events, state)
                succeeded += 1
            else:
                logger.warning(f"  ❌ 段 {idx+1} 失败")
                fallback = True
                break

        if not text.strip():
            logger.error("❌ ControlledWriter 完全失败")
            return ControlledWriteResult(
                text="",
                events=[],
                segments_used=0,
                segments_succeeded=0,
                fallback_used=fallback,
                execution_time=time.time() - start,
                original_text="",
                rewritten_text=None,
                rewrite_attempted=False,
                rewrite_failure_reason=None,
            )

        logger.info(f"✅ ControlledWriter 完成: {succeeded}/{segments} 段成功" +
                    (f" (降级)" if fallback else ""))

        final_text = text.strip()

        # ========== Phase 15.7-B1: 执行 Rewrite（仅观察） ==========
        # ========== Phase 15.8 Commit 1: Rewrite Selection ==========
        original_text = final_text
        execution_contract = getattr(contract, 'execution_contract', None)

        selection, rewritten_text = await self._select_rewrite(
            original_text=original_text,
            execution_contract=execution_contract,
        )

        logger.info(
            "[15.8-C1] selection: source=%s reason=%s available=%s safe=%s attempted=%s",
            selection.selected_source,
            selection.selection_reason,
            selection.rewrite_available,
            selection.structural_safe,
            selection.rewrite_attempted,
        )

        # Phase 15.8 Commit 3B: 根据 selection 决定 final_text
        if selection.selected_source == "rewritten" and rewritten_text:
            final_text = rewritten_text
            logger.info(
                "[15.8-C3B] final_text=rewritten (len=%d)", len(final_text)
            )
        else:
            final_text = original_text
            logger.info(
                "[15.8-C3B] final_text=original (reason=%s)",
                selection.selection_reason,
            )
            
        return ControlledWriteResult(
            text=final_text,
            events=events,
            segments_used=segments,
            segments_succeeded=succeeded,
            fallback_used=fallback,
            execution_time=time.time() - start,
            original_text=original_text,
            rewritten_text=rewritten_text,           # 双轨观察保留
            rewrite_attempted=selection.rewrite_attempted,
            rewrite_failure_reason=selection.rewrite_failure_reason,
            selection=selection,
        )
        # ======================================================================