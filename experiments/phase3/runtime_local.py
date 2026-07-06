# experiments/phase3/runtime_local.py
"""
Narrative Runtime - 优化版（增强错误恢复、自适应分段）
"""

import logging
import re
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from openai import AsyncOpenAI
import httpx

from experiments.phase1.planning_contract_local import PlanningContract, ExecutionUnit

logger = logging.getLogger(__name__)


@dataclass
class SegmentResult:
    text: str
    events: List[Dict]
    success: bool
    retry_count: int = 0
    error: Optional[str] = None
    fallback_used: bool = False  # 是否使用了降级


@dataclass
class RuntimeResult:
    full_text: str
    all_events: List[Dict]
    final_state: Dict[str, Any]
    segments: List[SegmentResult]
    total_segments: int
    successful_segments: int
    execution_time: float = 0.0
    fallback_used: bool = False


class NarrativeRuntime:
    def __init__(
        self,
        llm_api_base: str = "http://localhost:8082/v1",
        llm_model: str = "Qwen3-32B-Q5_K_M",
        max_retries_per_segment: int = 2,
        default_segments: int = 2,
        enable_fallback: bool = True,
    ):
        self.llm_api_base = llm_api_base
        self.llm_model = llm_model
        self.max_retries_per_segment = max_retries_per_segment
        self.default_segments = default_segments
        self.enable_fallback = enable_fallback
        
        self.current_state: Dict[str, Any] = {}
        self.all_events: List[Dict] = []
        self.all_text: str = ""
        self.segment_results: List[SegmentResult] = []
        self.fallback_used = False

    def schedule_segments(self, contract: PlanningContract, segments_hint: Optional[int] = None) -> List[List[ExecutionUnit]]:
        """
        智能分段调度
        - 如果 segments_hint 为 None 或 0，则根据单元数量自动决定
        - 否则使用给定的段数
        """
        units = contract.execution.units
        total = len(units)
        
        if segments_hint is not None and segments_hint > 0:
            # 使用指定的段数
            segments = segments_hint
        else:
            # 自动分段
            if total <= 2:
                segments = 1
            elif total <= 4:
                segments = 2
            elif total <= 6:
                segments = 3
            else:
                segments = 4
        
        if segments == 1:
            return [units]
        
        # 平均分配
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

    def _build_context_summary(self) -> str:
        if not self.all_events:
            return "尚未发生任何事件。"
        summaries = []
        for evt in self.all_events[-5:]:
            evt_type = evt.get("type", "")
            if evt_type == "plot_flag_set":
                summaries.append(f"📌 {evt.get('flag')} = {evt.get('value')}")
            elif evt_type == "item_acquire":
                summaries.append(f"🎒 {evt.get('actor')} 获得 {evt.get('item')}")
            elif evt_type == "location_enter":
                summaries.append(f"📍 {evt.get('actor')} 进入 {evt.get('location')}")
            elif evt_type == "realm_upgrade":
                summaries.append(f"⚡ {evt.get('actor')} 突破到 {evt.get('to_major_realm')}{evt.get('to_minor_stage')}层")
            elif evt_type == "relationship_change":
                summaries.append(f"👥 {evt.get('from_char')} 与 {evt.get('to_char')} 关系变化 {evt.get('delta')}")
        return "\n".join(summaries) if summaries else "无重大事件。"

    def _build_state_summary(self) -> str:
        if not self.current_state:
            return "初始状态"
        lines = []
        chars = self.current_state.get("characters", {})
        for name, info in chars.items():
            hp = info.get("hp", "?")
            realm = info.get("realm", "?")
            level = info.get("level", 1)
            location = info.get("location", "未知")
            lines.append(f"  {name}: 境界={realm}{level}层, HP={hp}, 位置={location}")
        return "\n".join(lines) if lines else "无角色信息"

    def _build_segment_prompt(
        self,
        contract: PlanningContract,
        segment_units: List[ExecutionUnit],
        segment_idx: int,
        total_segments: int,
        is_retry: bool = False,
        is_fallback: bool = False,
    ) -> str:
        lines = []
        if is_fallback:
            lines.append("⚠️ 这是降级模式：由于分段执行失败，现在改为一次性生成完整场景。")
            lines.append("")
        else:
            lines.append(f"请写一段场景正文（约 200-300 字）。这是第 {segment_idx + 1}/{total_segments} 段。")
            lines.append("")
        
        lines.append("【场景总目标】")
        lines.append(f"目标：{contract.intent.goal}")
        lines.append(f"冲突：{contract.intent.conflict}")
        lines.append("")
        
        if not is_fallback and self.all_text:
            lines.append("【上一段结尾】")
            lines.append(self.all_text[-300:])
            lines.append("请自然衔接上一段结尾。")
            lines.append("")
        
        if self.all_events:
            lines.append("【已完成的事件】")
            lines.append(self._build_context_summary())
            lines.append("")
        
        if self.current_state:
            lines.append("【当前世界状态】")
            lines.append(self._build_state_summary())
            lines.append("")
        
        lines.append("【本段必须完成的执行单元】")
        for unit in segment_units:
            lines.append(f"- {unit.description}")
        lines.append("")
        
        if contract.constraints:
            lines.append("【硬性约束】")
            for c in contract.constraints:
                if c.type == "required":
                    lines.append(f"  ✅ 必须发生：{c.target}")
                elif c.type == "forbidden":
                    lines.append(f"  ❌ 禁止发生：{c.target}")
            lines.append("")
        
        if contract.observables.state_changes:
            lines.append("【必须记录的状态变化】")
            for change in contract.observables.state_changes:
                if change.type == "plot_flag":
                    lines.append(f"  - plot_flag: {change.name} = {change.value}")
                    lines.append(f"    事件格式: {{'type': 'plot_flag_set', 'flag': '{change.name}', 'value': {change.value}}}")
                elif change.type == "inventory":
                    lines.append(f"  - inventory: {change.actor} 获得 {change.item}")
                    lines.append(f"    事件格式: {{'type': 'item_acquire', 'actor': '{change.actor}', 'item': '{change.item}'}}")
                elif change.type == "relationship":
                    lines.append(f"  - relationship: {change.from_char} → {change.to_char} 变化 {change.delta}")
                    lines.append(f"    事件格式: {{'type': 'relationship_change', 'from_char': '{change.from_char}', 'to_char': '{change.to_char}', 'delta': {change.delta}}}")
            lines.append("")
        
        if is_retry:
            lines.append("⚠️ 上一轮验证失败，请重新生成本段。")
            lines.append("确保完成所有执行单元，并输出有效的 JSON。")
            lines.append("")
        
        if is_fallback:
            lines.append("【注意】请一次性生成完整的场景正文（约 500-800 字），覆盖所有执行单元。")
            lines.append("")
        
        lines.append("【输出格式】")
        lines.append('{"scene_text": "...", "events": [{"type": "...", ...}]}')
        lines.append("只输出 JSON，不要有任何额外文本。")
        
        return "\n".join(lines)

    def _verify_segment(self, text: str, segment_units: List[ExecutionUnit]) -> bool:
        if not segment_units:
            return True
        for unit in segment_units:
            keywords = re.findall(r'[\u4e00-\u9fff]{2,4}', unit.description)
            if not keywords:
                keywords = [unit.description[:6]]
            if not any(kw in text for kw in keywords):
                return False
        return True

    def _apply_events(self, events: List[Dict]) -> Dict:
        state = self.current_state.copy()
        if not state:
            state = {
                "characters": {"林逸": {"hp": 100, "realm": "炼气", "level": 1, "inventory": []}},
                "relationships": {},
                "global_flags": {}
            }
        for evt in events:
            evt_type = evt.get("type", "")
            if evt_type == "plot_flag_set":
                flag = evt.get("flag")
                value = evt.get("value", True)
                if flag:
                    state.setdefault("global_flags", {})[flag] = value
            elif evt_type == "item_acquire":
                actor = evt.get("actor")
                item = evt.get("item")
                if actor and item:
                    if actor not in state["characters"]:
                        state["characters"][actor] = {"inventory": []}
                    if "inventory" not in state["characters"][actor]:
                        state["characters"][actor]["inventory"] = []
                    state["characters"][actor]["inventory"].append(item)
            elif evt_type == "location_enter":
                actor = evt.get("actor")
                location = evt.get("location")
                if actor and location:
                    if actor not in state["characters"]:
                        state["characters"][actor] = {}
                    state["characters"][actor]["location"] = location
            elif evt_type == "realm_upgrade":
                actor = evt.get("actor")
                realm = evt.get("to_major_realm")
                level = evt.get("to_minor_stage")
                if actor and realm:
                    if actor not in state["characters"]:
                        state["characters"][actor] = {}
                    state["characters"][actor]["realm"] = realm
                    if level:
                        state["characters"][actor]["level"] = level
            elif evt_type == "relationship_change":
                from_char = evt.get("from_char")
                to_char = evt.get("to_char")
                delta = evt.get("delta", 0)
                if from_char and to_char:
                    key = f"{from_char}|{to_char}"
                    state.setdefault("relationships", {})[key] = state["relationships"].get(key, 0) + delta
            elif evt_type == "hp_changed":
                actor = evt.get("actor")
                new_hp = evt.get("new_hp")
                if actor and new_hp is not None:
                    if actor not in state["characters"]:
                        state["characters"][actor] = {}
                    state["characters"][actor]["hp"] = new_hp
        return state

    async def _call_llm(self, prompt: str, max_tokens: int = 2048) -> str:
        client = AsyncOpenAI(
            api_key="not-needed",
            base_url=self.llm_api_base,
            timeout=httpx.Timeout(600.0, connect=30.0)
        )
        response = await client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def _parse_json(self, text: str) -> Dict:
        if not text:
            return {}
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        return {}

    async def _execute_segment(
        self,
        contract: PlanningContract,
        segment_units: List[ExecutionUnit],
        segment_idx: int,
        total_segments: int,
    ) -> SegmentResult:
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries_per_segment:
            is_retry = retry_count > 0
            prompt = self._build_segment_prompt(
                contract=contract,
                segment_units=segment_units,
                segment_idx=segment_idx,
                total_segments=total_segments,
                is_retry=is_retry,
            )
            try:
                response = await self._call_llm(prompt, max_tokens=2048)
                data = self._parse_json(response)
                text = data.get("scene_text", "")
                events = data.get("events", [])
                if not text or len(text.strip()) < 20:
                    raise ValueError("生成的文本过短")
                if self._verify_segment(text, segment_units):
                    logger.info(f"  ✅ 段 {segment_idx+1} 成功 (重试 {retry_count} 次)")
                    return SegmentResult(text=text, events=events, success=True, retry_count=retry_count)
                else:
                    raise ValueError("验证失败：未完成所有执行单元")
            except Exception as e:
                last_error = str(e)
                retry_count += 1
                logger.warning(f"  ⚠️ 段 {segment_idx+1} 失败 (尝试 {retry_count}/{self.max_retries_per_segment+1}): {e}")
        
        # 重试次数用尽，尝试降级
        if self.enable_fallback:
            logger.warning(f"  🔄 段 {segment_idx+1} 重试失败，尝试降级到单次生成")
            fallback_prompt = self._build_segment_prompt(
                contract=contract,
                segment_units=contract.execution.units,
                segment_idx=0,
                total_segments=1,
                is_retry=False,
                is_fallback=True,
            )
            try:
                fallback_response = await self._call_llm(fallback_prompt, max_tokens=4096)
                fallback_data = self._parse_json(fallback_response)
                fallback_text = fallback_data.get("scene_text", "")
                fallback_events = fallback_data.get("events", [])
                if fallback_text and len(fallback_text.strip()) > 50:
                    logger.info(f"  ✅ 降级成功，生成完整场景")
                    # 清空已有内容，使用降级结果
                    self.all_text = fallback_text
                    self.all_events = fallback_events
                    self.current_state = self._apply_events(fallback_events)
                    # 标记使用了降级
                    self.fallback_used = True
                    return SegmentResult(
                        text=fallback_text,
                        events=fallback_events,
                        success=True,
                        retry_count=retry_count,
                        error=None,
                        fallback_used=True,
                    )
                else:
                    logger.error("降级生成失败，返回空结果")
            except Exception as e:
                logger.error(f"降级生成异常: {e}")
        
        return SegmentResult(
            text="",
            events=[],
            success=False,
            retry_count=retry_count,
            error=last_error,
            fallback_used=False,
        )

    async def execute(self, contract: PlanningContract, segments_hint: Optional[int] = None) -> RuntimeResult:
        start_time = time.time()
        self.current_state = {}
        self.all_events = []
        self.all_text = ""
        self.segment_results = []
        self.fallback_used = False
        
        segment_units_list = self.schedule_segments(contract, segments_hint)
        total_segments = len(segment_units_list)
        logger.info(f"🚀 Runtime 启动: {len(contract.execution.units)} 个执行单元 → {total_segments} 段 (hint={segments_hint})")
        
        successful_segments = 0
        for idx, units in enumerate(segment_units_list):
            logger.info(f"  📝 执行段 {idx+1}/{total_segments} ({len(units)} 个单元)")
            result = await self._execute_segment(
                contract=contract,
                segment_units=units,
                segment_idx=idx,
                total_segments=total_segments,
            )
            self.segment_results.append(result)
            if result.success:
                successful_segments += 1
                # 如果是降级模式，已经覆盖了全部内容，不再继续
                if result.fallback_used:
                    logger.info("  ℹ️ 降级模式已生成完整场景，结束执行")
                    break
                self.all_text += result.text + "\n\n"
                self.all_events.extend(result.events)
                self.current_state = self._apply_events(result.events)
            else:
                logger.warning(f"  ❌ 段 {idx+1} 失败: {result.error}")
                # 如果失败且没有降级，直接停止
                break
        
        execution_time = time.time() - start_time
        logger.info(f"✅ Runtime 完成: {successful_segments}/{total_segments} 段成功，耗时 {execution_time:.2f}s")
        return RuntimeResult(
            full_text=self.all_text,
            all_events=self.all_events,
            final_state=self.current_state,
            segments=self.segment_results,
            total_segments=total_segments,
            successful_segments=successful_segments,
            execution_time=execution_time,
            fallback_used=self.fallback_used,
        )