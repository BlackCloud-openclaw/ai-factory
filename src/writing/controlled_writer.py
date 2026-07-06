"""
Controlled Writer - 产品化增量执行服务

将 Phase 2/3 验证有效的增量执行能力封装为稳定的服务。
"""

import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from openai import AsyncOpenAI
import httpx

from src.writing.planning_contract import PlanningContract, ExecutionUnit
from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ControlledWriteResult:
    """受控写入结果"""
    text: str
    events: List[Dict]
    segments_used: int
    segments_succeeded: int
    fallback_used: bool
    execution_time: float


class ControlledWriter:
    """
    受控写入器 - 产品级增量执行引擎
    
    设计原则：
    1. 稳定 > 速度：优先保证输出，再考虑效率
    2. 降级兜底：任何失败都回退到单次执行
    3. 状态隔离：每次执行独立，不污染全局
    """
    
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
        """根据执行单元数量决定分段数"""
        total = len(units)
        if total <= 2:
            return 1
        elif total <= 4:
            return 2
        elif total <= 6:
            return 3
        else:
            return 4
    
    def _split_units(self, units: List[ExecutionUnit], segments: int) -> List[List[ExecutionUnit]]:
        """将执行单元分配到各段"""
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
    ) -> str:
        """构建段 Prompt"""
        lines = []
        
        if is_fallback:
            lines.append("⚠️ 降级模式：一次性生成完整场景。")
            lines.append("")
        else:
            lines.append(f"请写一段场景正文（约 200-300 字）。这是第 {segment_idx + 1}/{total_segments} 段。")
            lines.append("")
        
        lines.append("【场景总目标】")
        lines.append(f"目标：{contract.intent.goal}")
        lines.append(f"冲突：{contract.intent.conflict}")
        lines.append("")
        
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
        
        if contract.constraints:
            lines.append("【约束】")
            for c in contract.constraints:
                if c.type == "required":
                    lines.append(f"  ✅ 必须：{c.target}")
                elif c.type == "forbidden":
                    lines.append(f"  ❌ 禁止：{c.target}")
            lines.append("")
        
        if is_retry:
            lines.append("⚠️ 重试：请修正上一轮的错误。")
            lines.append("")
        
        if is_fallback:
            lines.append("【要求】一次性生成 500-800 字完整场景。")
            lines.append("")
        
        lines.append("【格式】")
        lines.append('{"scene_text": "...", "events": [...]}')
        lines.append("只输出 JSON。")
        
        return "\n".join(lines)
    
    def _verify_segment(self, text: str, units: List[ExecutionUnit]) -> bool:
        """验证段是否完成"""
        if not units:
            return True
        for unit in units:
            keywords = re.findall(r'[\u4e00-\u9fff]{2,4}', unit.description)
            if not keywords:
                keywords = [unit.description[:6]]
            if not any(kw in text for kw in keywords):
                return False
        return True
    
    def _apply_events(self, events: List[Dict], state: Dict) -> Dict:
        """应用事件更新状态"""
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
    
    async def _call_llm(self, prompt: str, max_tokens: int = 2048) -> str:
        """调用 LLM"""
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
        )
        return response.choices[0].message.content or ""
    
    def _parse_json(self, text: str) -> Dict:
        """解析 JSON 响应"""
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
        units: List[ExecutionUnit],
        idx: int,
        total: int,
        previous_text: str,
        previous_events: List[Dict],
        current_state: Dict,
    ) -> Tuple[str, List[Dict], bool]:
        """执行单个段（含重试）"""
        for attempt in range(self.max_retries_per_segment + 1):
            is_retry = attempt > 0
            prompt = self._build_segment_prompt(
                contract=contract,
                segment_units=units,
                segment_idx=idx,
                total_segments=total,
                previous_text=previous_text,
                previous_events=previous_events,
                current_state=current_state,
                is_retry=is_retry,
            )
            try:
                response = await self._call_llm(prompt, max_tokens=2048)
                data = self._parse_json(response)
                text = data.get("scene_text", "")
                events = data.get("events", [])
                if not text or len(text.strip()) < 20:
                    raise ValueError("生成文本过短")
                if self._verify_segment(text, units):
                    logger.info(f"  ✅ 段 {idx+1} 成功 (尝试 {attempt+1})")
                    return text, events, True
                else:
                    raise ValueError("验证失败")
            except Exception as e:
                logger.warning(f"  ⚠️ 段 {idx+1} 失败 (尝试 {attempt+1}): {e}")
                if attempt == self.max_retries_per_segment:
                    if self.enable_fallback:
                        logger.info(f"  🔄 段 {idx+1} 降级到单次生成")
                        fallback_prompt = self._build_segment_prompt(
                            contract=contract,
                            segment_units=contract.execution.units,
                            idx=0,
                            total=1,
                            previous_text="",
                            previous_events=[],
                            current_state={},
                            is_retry=False,
                            is_fallback=True,
                        )
                        try:
                            fb_response = await self._call_llm(fallback_prompt, max_tokens=4096)
                            fb_data = self._parse_json(fb_response)
                            fb_text = fb_data.get("scene_text", "")
                            fb_events = fb_data.get("events", [])
                            if fb_text and len(fb_text.strip()) > 50:
                                logger.info(f"  ✅ 降级成功")
                                return fb_text, fb_events, True
                        except Exception as fb_e:
                            logger.error(f"  ❌ 降级失败: {fb_e}")
        return "", [], False
    
    async def execute(self, contract: PlanningContract) -> ControlledWriteResult:
        """
        执行受控写入
        
        Args:
            contract: Planning Contract
            
        Returns:
            ControlledWriteResult
        """
        import time
        start = time.time()
        
        # 1. 决定分段
        units = contract.execution.units
        segments = self._determine_segments(units)
        segment_units = self._split_units(units, segments)
        
        logger.info(f"📝 ControlledWriter: {len(units)} 单元 → {segments} 段")
        
        # 2. 状态
        text = ""
        events = []
        state = {}
        succeeded = 0
        fallback = False
        
        # 3. 逐段执行
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
                # 检查是否使用了降级（降级会生成完整文本，直接结束）
                if seg_text and len(seg_text.strip()) > 100:
                    # 降级成功，已经生成了完整场景
                    text = seg_text
                    events = seg_events
                    state = self._apply_events(seg_events, state)
                    fallback = True
                    succeeded = 1
                break
        
        # 4. 如果没有任何成功，返回空
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
        
        # 5. 如果部分成功但使用了降级，正常返回
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