"""
Controlled Writer - 产品化增量执行服务

稳定性优化 v2：强制 JSON 输出，Pydantic 验证，针对性重试，降级兜底。
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
from src.config import settings

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
    def __init__(
        self,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        max_retries_per_segment: int = 2,
        enable_fallback: bool = True,
    ):
        self.api_base = api_base or settings.llm_api_url
        self.model = model or getattr(settings, 'llm_writing_model', 'Qwen3-32B-Q5_K_M')
        self.max_retries_per_segment = max_retries_per_segment
        self.enable_fallback = enable_fallback

    def _determine_segments(self, units: List[ExecutionUnit]) -> int:
        """
        分段策略：减少分段数，让每段篇幅更长
        - 1-4 个单元 → 1 段（整场景）
        - 5-8 个单元 → 2 段
        - 9+ 个单元 → 3 段
        """
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
        contract: PlanningContract,
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

        if is_fallback:
            lines.append("⚠️ 降级模式：一次性生成完整场景，约 800-1200 字。")
            lines.append("")
        else:
            lines.append(f"请写一段场景正文（约 400-600 字）。这是第 {segment_idx + 1}/{total_segments} 段。")
            lines.append("")

        lines.append("【场景总目标】")
        lines.append(f"目标：{contract.intent.goal}")
        lines.append(f"冲突：{contract.intent.conflict}")
        lines.append("")

        # ========== v2.1: 注入 Scene Specification 强制渲染指令（优先级最高） ==========
        if contract.scene_spec:
            spec = contract.scene_spec
            
            lines.append("=" * 60)
            lines.append("🎯 [v2.1 最高优先级] 场景渲染规格 - 必须严格遵循")
            lines.append("=" * 60)
            lines.append("")
            lines.append("⚠️ 以下规格的优先级高于下方的【本段任务】")
            lines.append("⚠️ 如果执行单元与规格冲突，优先满足规格")
            lines.append("")
            
            # 1. 世界事实（强制渲染）
            lines.append("【🌍 世界事实（必须在正文中直接渲染，不可省略）】")
            lines.append(f"  地点：{spec.world.get('location', '未指定')}")
            lines.append(f"  时间：{spec.world.get('time', '未指定')}")
            lines.append(f"  氛围：{spec.world.get('atmosphere', '未指定')}")
            if spec.world.get('sensory'):
                lines.append(f"  感官细节：{', '.join(spec.world['sensory'])}")
            lines.append("")
            
            # 2. 情绪轨迹（强制引导）
            lines.append("【❤️ 读者情绪轨迹（必须让读者经历以下三段式变化）】")
            lines.append(f"  开头 → 读者应感到：{spec.reader_emotion.get('begin', '未指定')}")
            lines.append(f"  中间 → 读者应感到：{spec.reader_emotion.get('middle', '未指定')}")
            lines.append(f"  结尾 → 读者应感到：{spec.reader_emotion.get('end', '未指定')}")
            lines.append("")
            
            # 3. 叙事功能（结构约束）
            lines.append("【📖 叙事功能】")
            lines.append(f"  场景功能：{spec.narrative_function}")
            lines.append(f"  功能含义：{self._get_function_meaning(spec.narrative_function)}")
            lines.append("")
            
            # 4. 视角锚定（强制）
            lines.append("【👁️ 视角约束】")
            lines.append(f"  全程从「{spec.pov}」的视角叙述")
            lines.append("  不要出现其他角色的内心独白")
            lines.append("")
            
            # 5. 渲染规则
            lines.append("【⚠️ 渲染规则（必须遵守）】")
            lines.append("1. 必须在正文中直接描写上述【世界事实】中的所有元素")
            lines.append("2. 必须让读者经历【情绪轨迹】中的三段式变化")
            lines.append("3. 场景结构必须符合【叙事功能】的指导")
            lines.append("4. 只从【视角约束】指定的角色视角描述事件")
            lines.append("5. 不要解释或说明这些规格，直接用叙事文本呈现")
            lines.append("6. 不要使用「推进主线剧情」等占位符，直接用具体情节推进")
            lines.append("")
            lines.append("=" * 60)
            lines.append("")

        # 继续原有内容
        if not is_fallback and previous_text:
            lines.append("【上一段结尾】")
            lines.append(previous_text[-300:])
            lines.append("请自然衔接。")
            lines.append("")

        if previous_events:
            lines.append("【已完成事件】")
            for evt in previous_events[-5:]:
                if evt.get("type") == "plot_flag_set":
                    lines.append(f"  📌 {evt.get('flag')} = {evt.get('value')}")
                elif evt.get("type") == "item_acquire":
                    lines.append(f"  🎒 {evt.get('actor')} 获得 {evt.get('item')}")
                elif evt.get("type") == "location_enter":
                    lines.append(f"  📍 {evt.get('actor')} 进入 {evt.get('location')}")
            lines.append("")

        if current_state:
            lines.append("【当前状态】")
            chars = current_state.get("characters", {})
            for name, info in chars.items():
                hp = info.get("hp", "?")
                realm = info.get("realm", "?")
                level = info.get("level", 1)
                lines.append(f"  {name}: {realm}{level}层, HP={hp}")
            lines.append("")

        lines.append("【本段任务】")
        for unit in segment_units:
            lines.append(f"- {unit.description}")
        lines.append("")

        lines.append("【写作要求】")
        lines.append("1. 每个执行单元需要展开 100-150 字的细节描写")
        lines.append("2. 包含场景氛围、角色情绪、对话和动作的交织")
        lines.append("3. 不要只罗列事件，要用情节自然地展现")
        lines.append("4. 确保段落之间节奏连贯，有适当的铺垫和余韵")
        lines.append("5. 如果上一段结尾有未完成的对话或动作，优先承接")
        lines.append("")

        if contract.constraints:
            lines.append("【约束】")
            for c in contract.constraints:
                if c.type == "required":
                    lines.append(f"  ✅ 必须：{c.target}")
                elif c.type == "forbidden":
                    lines.append(f"  ❌ 禁止：{c.target}")
            lines.append("")

        if is_retry:
            lines.append("⚠️ 重试：请修正上一轮的错误，确保完成所有执行单元。")
            lines.append("")

        if is_fallback:
            lines.append("【要求】一次性生成 800-1200 字完整场景。")
            lines.append("")

        if error_hint:
            lines.append(error_hint)
            lines.append("")

        lines.append("【格式】")
        lines.append('{"scene_text": "...", "events": [...]}')
        lines.append("只输出 JSON。")

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
        """解析 JSON 并验证结构"""
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

    async def _call_llm(self, prompt: str, max_tokens: int = 4096) -> str:
        """调用 LLM，强制 JSON 输出"""
        client = AsyncOpenAI(
            api_key="not-needed",
            base_url=self.api_base,
            timeout=httpx.Timeout(600.0, connect=30.0)
        )
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content or ""

    async def _execute_segment(
        self,
        contract: PlanningContract,
        units: List[ExecutionUnit],
        idx: int,
        total: int,
        previous_text: str,
        previous_events: List[Dict],
        current_state: Dict,
    ) -> Tuple[str, List[Dict], bool]:
        for attempt in range(self.max_retries_per_segment + 1):
            is_retry = attempt > 0
            error_hint = ""
            
            if attempt == 1:
                error_hint = "\n\n⚠️ 格式错误：请只输出有效的 JSON，确保包含 scene_text（至少200字）和 events 数组。"
            elif attempt == 2:
                error_hint = "\n\n⚠️ 输出不完整：请生成完整的 JSON，scene_text 长度至少 300 字，并包含所有必要的执行单元。"

            prompt = self._build_segment_prompt(
                contract=contract,
                segment_units=units,
                segment_idx=idx,
                total_segments=total,
                previous_text=previous_text,
                previous_events=previous_events,
                current_state=current_state,
                is_retry=is_retry,
                error_hint=error_hint,
            )

            try:
                max_tokens = 4096 if attempt > 1 else 2048
                response = await self._call_llm(prompt, max_tokens=max_tokens)
                validated = self._parse_and_validate(response)

                if validated:
                    text = validated.scene_text
                    events = validated.events
                    if self._verify_segment(text, units):
                        logger.info(f"  ✅ 段 {idx+1} 成功 (尝试 {attempt+1}, 字数 {len(text)})")
                        return text, events, True
                    else:
                        logger.warning(f"  ⚠️ 段 {idx+1} 验证失败 (尝试 {attempt+1})")
                else:
                    logger.warning(f"  ⚠️ 段 {idx+1} 解析失败 (尝试 {attempt+1})")

            except Exception as e:
                logger.warning(f"  ⚠️ 段 {idx+1} 异常 (尝试 {attempt+1}): {e}")

        # 降级
        if self.enable_fallback:
            logger.warning(f"  🔄 段 {idx+1} 降级到单次生成")
            fallback_prompt = self._build_segment_prompt(
                contract=contract,
                segment_units=contract.execution.units,
                segment_idx=0,
                total_segments=1,
                previous_text="",
                previous_events=[],
                current_state={},
                is_retry=False,
                is_fallback=True,
                error_hint="\n\n⚠️ 降级模式：请一次性生成完整场景，长度至少 500 字。"
            )
            try:
                fb_response = await self._call_llm(fallback_prompt, max_tokens=4096)
                validated = self._parse_and_validate(fb_response)
                if validated and len(validated.scene_text.strip()) > 300:
                    logger.info(f"  ✅ 降级成功 (字数 {len(validated.scene_text)})")
                    return validated.scene_text, validated.events, True
            except Exception as e:
                logger.error(f"  ❌ 降级失败: {e}")

        return "", [], False

    async def execute(self, contract: PlanningContract) -> ControlledWriteResult:
        start = time.time()

        units = contract.execution.units
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
                if seg_text and len(seg_text.strip()) > 300:
                    text = seg_text
                    events = seg_events
                    state = self._apply_events(seg_events, state)
                    fallback = True
                    succeeded = 1
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

        return ControlledWriteResult(
            text=text.strip(),
            events=events,
            segments_used=segments,
            segments_succeeded=succeeded,
            fallback_used=fallback,
            execution_time=time.time() - start,
        )
            
    def _get_function_meaning(self, function: str) -> str:
        meanings = {
            "introduce_mystery": "留下谜团，不给出答案，结尾产生悬念或疑问",
            "escalate": "提升冲突，压力增大，局势紧张",
            "reveal_truth": "揭示关键信息，让读者感到震惊或恍然大悟",
            "release_tension": "缓解紧张情绪，提供情感喘息空间",
            "transition": "自然过渡，平稳衔接前后情节",
            "foreshadow": "埋下伏笔，暗示未来事件，不要明说",
        }
        return meanings.get(function, "推进场景叙事")