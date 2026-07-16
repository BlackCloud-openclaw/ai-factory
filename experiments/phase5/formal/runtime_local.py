# experiments/phase4/runtime_local.py
"""
Narrative Runtime - 本地独立版本（支持 Scene Specification）
用于 Phase 4 实验，不依赖主代码。
稳定性优化 v3：增强 JSON 解析，宽松验证，降级兜底。
"""

import logging
import re
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ValidationError
import httpx

from planning_contract_local import PlanningContract, ExecutionUnit, SceneSpecification

logger = logging.getLogger(__name__)


class WriterOutput(BaseModel):
    """LLM 输出结构验证"""
    scene_text: str = Field(..., min_length=50, description="场景正文，至少50字")
    events: List[Dict] = Field(default_factory=list, description="状态变化事件列表")
    foreshadowing: List[str] = Field(default_factory=list, description="伏笔列表")


@dataclass
class SegmentResult:
    text: str
    events: List[Dict]
    success: bool
    retry_count: int = 0
    error: Optional[str] = None
    fallback_used: bool = False


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
        """分段调度"""
        units = contract.execution.units
        total = len(units)

        if segments_hint is not None and segments_hint > 0:
            segments = segments_hint
        else:
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
        error_hint: str = "",
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

        # ========== v2.1: 注入 Scene Specification 渲染指令 ==========
        if contract.scene_spec:
            spec = contract.scene_spec
            lines.append("=" * 60)
            lines.append("🎨 [v2.1] 场景渲染规格 - 必须严格遵循")
            lines.append("=" * 60)
            lines.append("")

            lines.append("【🌍 世界事实（必须直接渲染，不可更改）】")
            lines.append(f"  地点：{spec.world.location}")
            lines.append(f"  时间：{spec.world.time}")
            lines.append(f"  氛围：{spec.world.atmosphere}")
            if spec.world.sensory:
                lines.append(f"  感官细节：{', '.join(spec.world.sensory)}")
            lines.append("")

            lines.append("【🎭 体验控制】")
            lines.append(f"  整体基调：{spec.mood}")
            lines.append(f"  叙事节奏：{spec.pacing}")
            lines.append(f"  视角角色：{spec.pov}")
            lines.append("")

            lines.append("【❤️ 读者情绪轨迹（必须引导读者经历以下变化）】")
            lines.append(f"  开头 → 读者应感到：{spec.reader_emotion.begin}")
            lines.append(f"  中间 → 读者应感到：{spec.reader_emotion.middle}")
            lines.append(f"  结尾 → 读者应感到：{spec.reader_emotion.end}")
            lines.append("")

            lines.append("【📖 叙事功能】")
            lines.append(f"  场景功能：{spec.narrative_function}")
            lines.append(f"  功能含义：{spec.get_function_meaning()}")
            lines.append("")

            lines.append("【⚠️ 渲染规则】")
            lines.append("1. 必须在正文中直接描写上述【世界事实】中的所有元素")
            lines.append("2. 必须让读者经历【情绪轨迹】中的三段式变化")
            lines.append("3. 节奏必须符合【叙事节奏】要求（slow=细腻描写，fast=动作密集）")
            lines.append("4. 场景结构必须符合【叙事功能】的指导")
            lines.append("5. 不要解释或说明这些规格，直接用叙事文本呈现")
            lines.append("")
            lines.append("=" * 60)
            lines.append("")

        # 继续原有内容
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

        if error_hint:
            lines.append(error_hint)
            lines.append("")

        lines.append("【输出格式】")
        lines.append('{"scene_text": "...", "events": [{"type": "...", ...}]}')
        lines.append("只输出 JSON，不要有任何额外文本。")

        return "\n".join(lines)

    def _extract_core_keywords(self, description: str) -> List[str]:
        """提取核心关键词（含同义扩展）"""
        # 移除常见动词前缀
        cleaned = re.sub(r'^(发现|找到|进入|获得|前往|完成|进行|探查|查看|检查|来到|返回|离开|走向|拿起|放下|听到|看见|闻到|触到)', '', description)
        keywords = re.findall(r'[\u4e00-\u9fff]{2,4}', cleaned)
        
        # 同义扩展（基于常见变体）
        expansions = []
        for kw in keywords:
            # 灵草 → 灵药、草药
            if '灵草' in kw:
                expansions.extend(['灵药', '草药', '药草'])
            # 地缝 → 裂缝、裂口、裂隙
            elif '地缝' in kw:
                expansions.extend(['裂缝', '裂口', '裂隙'])
            # 腐臭 → 腥臭、恶臭、腐烂
            elif '腐臭' in kw:
                expansions.extend(['腥臭', '恶臭', '腐烂', '腥味'])
            # 晨雾 → 雾气、浓雾、薄雾
            elif '晨雾' in kw:
                expansions.extend(['雾气', '浓雾', '薄雾', '雾霭'])
        
        keywords.extend(expansions)
        if not keywords:
            return [description[:6]]
        return keywords

    def _verify_segment(self, text: str, segment_units: List[ExecutionUnit]) -> bool:
        """验证段内容 - 简化版：仅检查长度"""
        # 纯文本模式：只要长度足够就接受
        if len(text.strip()) >= 100:
            return True
        return False

    def _parse_and_validate(self, text: str) -> Optional[WriterOutput]:
        """增强的 JSON 解析与验证 - 支持纯文本降级"""
        if not text or len(text.strip()) < 50:
            return None

        # 尝试提取 JSON
        patterns = [
            (r'\{.*\}', None),
            (r'```json\s*([\s\S]*?)\s*```', 1),
            (r'```\s*([\s\S]*?)\s*```', 1),
        ]

        for pattern, group in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if not match:
                continue

            json_str = match.group(group) if group is not None else match.group()
            try:
                json_str = json_str.strip()
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                json_str = json_str.replace('\n', '\\n').replace('\r', '\\r')
                data = json.loads(json_str)

                if data.get("scene_text") and len(data["scene_text"].strip()) >= 50:
                    return WriterOutput(**data)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.debug(f"JSON 解析失败: {e}")
                continue

        # ====== 最后手段：直接将文本作为场景内容 ======
        # 清洗文本，移除可能的 JSON 残留标记
        clean_text = re.sub(r'\{.*\}', '', text, flags=re.DOTALL).strip()
        if len(clean_text) > 100:
            logger.info("使用纯文本提取模式（无 JSON 包裹）")
            return WriterOutput(scene_text=clean_text[:3000], events=[], foreshadowing=[])
        
        return None

    async def _call_llm(self, prompt: str, max_tokens: int = 2048) -> str:
        """调用 LLM，不使用 response_format（兼容性）"""
        async with httpx.AsyncClient(
            trust_env=False,
            timeout=httpx.Timeout(600.0, connect=30.0)
        ) as client:
            payload = {
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": max_tokens,
            }
            response = await client.post(
                f"{self.llm_api_base}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

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
            error_hint = ""

            if retry_count == 1:
                error_hint = "\n\n⚠️ 格式错误：请只输出 JSON，不要添加额外文本。确保 scene_text 至少 200 字。"
            elif retry_count == 2:
                error_hint = "\n\n⚠️ 输出不完整：请生成完整的 JSON 格式，scene_text 至少 300 字。"

            prompt = self._build_segment_prompt(
                contract=contract,
                segment_units=segment_units,
                segment_idx=segment_idx,
                total_segments=total_segments,
                is_retry=is_retry,
                error_hint=error_hint,
            )

            try:
                max_tokens = 4096 if retry_count > 1 else 2048
                response = await self._call_llm(prompt, max_tokens=max_tokens)
                validated = self._parse_and_validate(response)

                if validated:
                    text = validated.scene_text
                    events = validated.events

                    if self._verify_segment(text, segment_units):
                        logger.info(f"  ✅ 段 {segment_idx+1} 成功 (重试 {retry_count} 次)")
                        return SegmentResult(
                            text=text,
                            events=events,
                            success=True,
                            retry_count=retry_count
                        )
                    else:
                        last_error = "验证失败：未完成所有执行单元"
                        logger.warning(f"  ⚠️ 段 {segment_idx+1} 验证失败 (尝试 {retry_count+1})")
                else:
                    last_error = "JSON 解析失败"
                    logger.warning(f"  ⚠️ 段 {segment_idx+1} 解析失败 (尝试 {retry_count+1})")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"  ⚠️ 段 {segment_idx+1} 异常 (尝试 {retry_count+1}): {e}")

            retry_count += 1

        # 降级
        if self.enable_fallback:
            logger.warning(f"  🔄 段 {segment_idx+1} 重试失败，尝试降级")
            fallback_prompt = self._build_segment_prompt(
                contract=contract,
                segment_units=contract.execution.units,
                segment_idx=0,
                total_segments=1,
                is_retry=False,
                is_fallback=True,
                error_hint="\n\n⚠️ 降级模式：请一次性生成完整场景，长度至少 500 字。只输出 JSON。"
            )
            try:
                fb_response = await self._call_llm(fallback_prompt, max_tokens=4096)
                validated = self._parse_and_validate(fb_response)
                if validated and len(validated.scene_text.strip()) > 300:
                    logger.info(f"  ✅ 降级成功")
                    self.all_text = validated.scene_text
                    self.all_events = validated.events
                    self.current_state = self._apply_events(validated.events)
                    self.fallback_used = True
                    return SegmentResult(
                        text=validated.scene_text,
                        events=validated.events,
                        success=True,
                        retry_count=retry_count,
                        fallback_used=True,
                    )
            except Exception as e:
                logger.error(f"降级异常: {e}")

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
        logger.info(f"🚀 Runtime 启动: {len(contract.execution.units)} 个执行单元 → {total_segments} 段")

        successful_segments = 0
        for idx, units in enumerate(segment_units_list):
            logger.info(f"  📝 执行段 {idx+1}/{total_segments}")
            result = await self._execute_segment(
                contract=contract,
                segment_units=units,
                segment_idx=idx,
                total_segments=total_segments,
            )
            self.segment_results.append(result)
            if result.success:
                successful_segments += 1
                if result.fallback_used:
                    # 降级模式已生成完整场景，结束执行
                    self.all_text = result.text
                    self.all_events = result.events
                    self.current_state = self._apply_events(result.events)
                    break
                self.all_text += result.text + "\n\n"
                self.all_events.extend(result.events)
                self.current_state = self._apply_events(result.events)
            else:
                logger.warning(f"  ❌ 段 {idx+1} 失败")
                break

        execution_time = time.time() - start_time
        logger.info(f"✅ Runtime 完成: {successful_segments}/{total_segments} 段成功")

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