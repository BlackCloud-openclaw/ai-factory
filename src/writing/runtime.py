"""
Narrative Runtime - 产品化增量执行引擎

将 Phase 2 验证有效的增量执行逻辑封装为可复用的 Runtime 组件。
"""

import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from openai import AsyncOpenAI
import httpx

from src.writing.planning_contract import PlanningContract, ExecutionUnit
from src.config import config

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class SegmentResult:
    """单个 Segment 的执行结果"""
    text: str
    events: List[Dict]
    success: bool
    retry_count: int = 0
    error: Optional[str] = None


@dataclass
class RuntimeResult:
    """Runtime 整体执行结果"""
    full_text: str
    all_events: List[Dict]
    final_state: Dict[str, Any]
    segments: List[SegmentResult]
    total_segments: int
    successful_segments: int
    execution_time: float = 0.0


# ============================================================================
# Runtime 核心类
# ============================================================================

class NarrativeRuntime:
    """
    叙事运行时引擎
    
    职责：
    1. 接收 PlanningContract
    2. 智能分段
    3. 增量执行
    4. 事件聚合
    5. 返回完整结果
    """
    
    def __init__(
        self,
        llm_api_base: Optional[str] = None,
        llm_model: Optional[str] = None,
        max_retries_per_segment: int = 2,
        default_segments: int = 2,
    ):
        self.llm_api_base = llm_api_base or getattr(config, 'llm_api_url', 'http://localhost:8082/v1')
        self.llm_model = llm_model or getattr(config, 'llm_writing_model', 'Qwen3-32B-Q5_K_M')
        self.max_retries_per_segment = max_retries_per_segment
        self.default_segments = default_segments
        
        # 状态
        self.current_state: Dict[str, Any] = {}
        self.all_events: List[Dict] = []
        self.all_text: str = ""
        self.segment_results: List[SegmentResult] = []
    
    # ========================================================================
    # 1. 分段调度器
    # ========================================================================
    
    def schedule_segments(self, contract: PlanningContract) -> List[List[ExecutionUnit]]:
        """
        根据执行单元数量决定分段策略
        
        策略：
        - 1-2 units → 1段
        - 3-4 units → 2段
        - 5-6 units → 3段
        - 7+ units → 4段
        """
        units = contract.execution.units
        total = len(units)
        
        if total <= 2:
            segments = 1
        elif total <= 4:
            segments = 2
        elif total <= 6:
            segments = 3
        else:
            segments = 4
        
        logger.info(f"📊 分段调度: {total} 个执行单元 → {segments} 段")
        
        # 分配单元到各段
        if segments == 1:
            return [units]
        
        # 尽量平均分配，每段至少1个单元
        base = total // segments
        remainder = total % segments
        
        result = []
        idx = 0
        for i in range(segments):
            count = base + (1 if i < remainder else 0)
            if count == 0:
                count = 1  # 确保每段至少有1个
            result.append(units[idx:idx + count])
            idx += count
        
        # 如果还有剩余（通常不会），加到最后一段
        if idx < total:
            result[-1].extend(units[idx:])
        
        return result
    
    # ========================================================================
    # 2. 上下文管理器
    # ========================================================================
    
    def _build_context_summary(self) -> str:
        """构建段间上下文摘要"""
        if not self.all_events:
            return "尚未发生任何事件。"
        
        summaries = []
        for evt in self.all_events[-5:]:  # 最近5个事件
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
        """构建当前状态摘要"""
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
    
    # ========================================================================
    # 3. Segment 执行器
    # ========================================================================
    
    def _build_segment_prompt(
        self,
        contract: PlanningContract,
        segment_units: List[ExecutionUnit],
        segment_idx: int,
        total_segments: int,
        is_retry: bool = False,
    ) -> str:
        """构建单个 Segment 的 Prompt"""
        lines = []
        lines.append(f"请写一段场景正文（约 200-300 字）。这是第 {segment_idx + 1}/{total_segments} 段。")
        lines.append("")
        lines.append("【场景总目标】")
        lines.append(f"目标：{contract.intent.goal}")
        lines.append(f"冲突：{contract.intent.conflict}")
        lines.append("")
        
        if self.all_text:
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
        
        # Observable Outcomes 提示
        if contract.observables.state_changes:
            lines.append("【必须记录的状态变化】")
            lines.append("请在生成的事件中记录以下状态变化：")
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
        
        lines.append("【输出格式】")
        lines.append('{"scene_text": "...", "events": [{"type": "...", ...}]}')
        lines.append("只输出 JSON，不要有任何额外文本。")
        
        return "\n".join(lines)
    
    def _verify_segment(self, text: str, segment_units: List[ExecutionUnit]) -> bool:
        """验证本段是否完成了分配的单元"""
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
        """将事件应用到当前状态"""
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
        """调用 LLM"""
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
        segment_units: List[ExecutionUnit],
        segment_idx: int,
        total_segments: int,
    ) -> SegmentResult:
        """执行单个 Segment（含重试）"""
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
                    return SegmentResult(
                        text=text,
                        events=events,
                        success=True,
                        retry_count=retry_count,
                    )
                else:
                    raise ValueError("验证失败：未完成所有执行单元")
                
            except Exception as e:
                last_error = str(e)
                retry_count += 1
                logger.warning(f"  ⚠️ 段 {segment_idx+1} 失败 (尝试 {retry_count}/{self.max_retries_per_segment+1}): {e}")
        
        # 重试次数用尽，返回失败结果
        return SegmentResult(
            text="",
            events=[],
            success=False,
            retry_count=retry_count,
            error=last_error,
        )
    
    # ========================================================================
    # 4. 主执行入口
    # ========================================================================
    
    async def execute(self, contract: PlanningContract) -> RuntimeResult:
        """
        执行完整的 Runtime 流程
        
        Args:
            contract: Planning Contract
            
        Returns:
            RuntimeResult: 包含完整文本、事件和状态
        """
        import time
        start_time = time.time()
        
        # 重置状态
        self.current_state = {}
        self.all_events = []
        self.all_text = ""
        self.segment_results = []
        
        # 1. 分段调度
        segment_units_list = self.schedule_segments(contract)
        total_segments = len(segment_units_list)
        
        logger.info(f"🚀 Runtime 启动: {len(contract.execution.units)} 个执行单元 → {total_segments} 段")
        
        # 2. 逐段执行
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
                self.all_text += result.text + "\n\n"
                self.all_events.extend(result.events)
                self.current_state = self._apply_events(result.events)
            else:
                logger.warning(f"  ❌ 段 {idx+1} 失败: {result.error}")
                # 如果某段失败，尝试用单次生成填充
                # 这里简单处理：记录错误，继续
                # 更好的处理：降级到单次生成
                break
        
        # 3. 如果所有段都成功，聚合结果
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
        )
    
    # ========================================================================
    # 5. 便捷方法：从 Contract 直接执行
    # ========================================================================
    
    @classmethod
    async def run(
        cls,
        contract: PlanningContract,
        llm_api_base: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> RuntimeResult:
        """静态方法：快速执行"""
        runtime = cls(llm_api_base=llm_api_base, llm_model=llm_model)
        return await runtime.execute(contract)