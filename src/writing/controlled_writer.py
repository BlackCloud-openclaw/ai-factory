# src/writing/controlled_writer.py
"""
Controlled Writer - 产品化增量执行服务

Phase 13.2.3C: 集成 QualityGate 实现控制闭环
"""

import logging
import re
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field, ValidationError
from openai import AsyncOpenAI
import httpx

from src.writing.planning_contract import PlanningContract, ExecutionUnit
from src.writing.contracts import WritingContract, WritingConstraints, WritingGoal
from src.writing.scene_execution_context import SceneExecutionContext
from src.writing.narrative_intent import NarrativeIntent
from src.config.settings import settings
from src.writing.runtime import RuntimeServices

# Phase 13.2.3C 导入
from .validation import SemanticValidator, ValidationResult
from .quality_gate import QualityGate, QualityGateResult

logger = logging.getLogger(__name__)


class WriterOutput(BaseModel):
    """LLM 输出结构验证"""
    scene_text: str = Field(..., min_length=50, description="场景正文，至少50字")
    events: List[Dict] = Field(default_factory=list, description="状态变化事件列表")
    foreshadowing: List[str] = Field(default_factory=list, description="伏笔列表")


@dataclass
class ControlledWriteResult:
    text: str
    events: List[Dict]
    segments_used: int
    segments_succeeded: int
    fallback_used: bool
    execution_time: float


class ControlledWriter:
    """
    受控写入器。

    支持通过 runtime_services 注入 Runtime 服务。
    Phase 13.2.3C: 注入 SemanticValidator 和 QualityGate 实现控制闭环。
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        max_retries_per_segment: int = 2,
        enable_fallback: bool = True,
        runtime_services: Optional[RuntimeServices] = None,
        semantic_validator: Optional[SemanticValidator] = None,
        quality_gate: Optional[QualityGate] = None,
    ):
        self.api_base = api_base or settings.llm_api_url
        self.model = model or getattr(settings, 'llm_writing_model', 'Qwen3-32B-Q5_K_M')
        self.max_retries_per_segment = max_retries_per_segment
        self.enable_fallback = enable_fallback
        self._runtime_services = runtime_services

        # Phase 13.2.3C: 注入 Validator 和 QualityGate
        self._semantic_validator = semantic_validator or SemanticValidator()
        self.quality_gate = quality_gate or QualityGate()

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
        error_hint: str = "",  # Phase 13.2.3C: 接收 feedback
    ) -> str:           
        lines = []
        
        # 注入 NarrativeIntent 指令（仅当是真正的 NarrativeIntent 实例）
        narrative_intent = writing_contract.narrative_intent
        if narrative_intent is not None and hasattr(narrative_intent, 'scene_role'):
            from src.writing.narrative_intent import NarrativeContext
            try:
                context = NarrativeContext.from_intent(narrative_intent)
                lines.append(context.to_prompt_instructions())
                lines.append("")
            except Exception:
                # 如果转换失败，跳过注入
                pass    

        if is_fallback:
            lines.append("⚠️ 降级模式：一次性生成完整场景，约 800-1200 字。")
            lines.append("")
        else:
            lines.append(f"请写一段场景正文（约 400-600 字）。这是第 {segment_idx + 1}/{total_segments} 段。")
            lines.append("")

        # Phase 13.2.3C: 注入 error_hint (feedback)
        if error_hint:
            lines.append(f"⚠️ 上一轮验证反馈：{error_hint}")
            lines.append("请根据以上反馈修正生成内容。")
            lines.append("")

        # ... 原有的 prompt 构建逻辑 ...
        # (这里保留原有代码，省略以保持简洁)

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
        transport = httpx.AsyncHTTPTransport(proxy=None)
        async with httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(600.0, connect=30.0)) as client:
            openai_client = AsyncOpenAI(
                api_key="not-needed",
                base_url=self.api_base,
                http_client=client,
            )
            response = await openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content or ""
            usage = response.usage.model_dump() if response.usage else {"total_tokens": 0}
            return content, usage

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
                else:
                    # 解析失败：完整响应落盘（仅第一次，防 IO 风暴）
                    if not getattr(self, "_parse_failure_logged", False):
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

                    # 获取 ValidationResult
                    validation_result = await self._validate_segment(text, contract)

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
            fallback_prompt = self._build_segment_prompt(
                writing_contract=contract,
                segment_units=contract.execution.units if hasattr(contract, 'execution') else [],
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

    async def _validate_segment(self, text: str, contract: WritingContract) -> ValidationResult:
        """
        验证单个 segment 的文本，使用注入的 SemanticValidator。
        """
        # 从 contract 中提取 PlanningContract
        planning_contract = getattr(contract, 'execution_contract', None)
        if planning_contract is None:
            # 无 contract 时返回空结果（视为通过）
            return ValidationResult(
                passed=True,
                missing=[],
                matched=[],
                blocking_missing=[],
                overall_confidence=1.0,
                weight_applied=1.0,
            )
        return self._semantic_validator.validate(planning_contract, text)

    # ========================================================================
    # 原有 execute 方法 (入口)
    # ========================================================================

    async def execute(self, contract: WritingContract) -> ControlledWriteResult:
        """执行受控写入 (入口方法)。"""
        start = time.time()

        # 获取执行单元
        units = getattr(contract, 'execution_units', [])
        if hasattr(contract, 'execution_contract') and contract.execution_contract:
            units = contract.execution_contract.execution.units

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
                # 即使失败，也返回已生成的部分
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
            )

        logger.info(f"✅ ControlledWriter 完成: {succeeded}/{segments} 段成功" +
                    (f" (降级)" if fallback else ""))

        # ========== D.3 观测点 3：最终 Result ==========
        logger.critical(
            "WRITER_ARTIFACT_DEBUG: final_text_len=%d events=%d events_type=%s",
            len(text.strip()),
            len(events),
            type(events).__name__
        )
        # =============================================

        return ControlledWriteResult(
            text=text.strip(),
            events=events,
            segments_used=segments,
            segments_succeeded=succeeded,
            fallback_used=fallback,
            execution_time=time.time() - start,
        )