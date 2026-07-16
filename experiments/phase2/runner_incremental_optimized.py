#!/usr/bin/env python3
"""
Phase 2 实验运行器 - 增量执行（Incremental Writer）优化版
改进点：
1. 段间状态传递增强
2. 段间验证与重试
3. 降级机制
4. 优化分段策略
"""

import asyncio
import json
import yaml
import logging
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import httpx
from openai import AsyncOpenAI

# 从 Phase 1 导入本地 Planning Contract
sys.path.insert(0, str(Path(__file__).parent.parent))
from phase1.planning_contract_local import (
    PlanningContract,
    Intent,
    Execution,
    ExecutionUnit,
    Observables,
    StateChange,
    Constraint,
    ContractMetadata,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phase2_incremental_optimized")

# ============================================================================
# 硬编码配置
# ============================================================================

LLM_API_BASE = "http://localhost:8082/v1"
LLM_MODEL = "Qwen3-32B-Q5_K_M"
LLM_TIMEOUT = 600

# ============================================================================
# 辅助函数
# ============================================================================

def load_config(path: Path) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def parse_json_response(text: str) -> Dict:
    """提取 JSON，确保返回字典"""
    if not text or not isinstance(text, str):
        return {}
    text = text.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            json_str = match.group()
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            try:
                result = json.loads(json_str)
                if isinstance(result, dict):
                    return result
            except:
                pass
    return {}

async def call_llm_with_retry(prompt: str, max_tokens: int = 4096, retries: int = 2) -> str:
    """带重试的 LLM 调用"""
    for attempt in range(retries + 1):
        try:
            client = AsyncOpenAI(
                api_key="not-needed",
                base_url=LLM_API_BASE,
                timeout=httpx.Timeout(LLM_TIMEOUT, connect=30.0)
            )
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            if content and len(content.strip()) > 10:
                return content
            logger.warning(f"LLM response too short (attempt {attempt+1})")
        except Exception as e:
            logger.warning(f"LLM call failed (attempt {attempt+1}): {e}")
            if attempt < retries:
                await asyncio.sleep(1)
    return ""

# ============================================================================
# Prompt 构建
# ============================================================================

def build_planner_prompt() -> str:
    return """
你是一位专业的小说规划师。请根据以下要求生成一个场景的 Planning Contract。

表示方式: action
信息密度: dense

输出格式必须是符合以下 Schema 的 JSON：

{
  "version": "1.0",
  "scene_id": "scene_001",
  "intent": {
    "goal": "场景目标（一句话）",
    "conflict": "核心冲突（一句话）",
    "expected_outcome": "预期结果（一句话）"
  },
  "execution": {
    "units": [
      {"id": "U1", "label": "action", "description": "执行步骤描述", "attributes": {}}
    ]
  },
  "observables": {
    "state_changes": [
      {"type": "plot_flag", "name": "flag_name", "value": true}
    ],
    "story_events": [],
    "narrative_flags": []
  },
  "constraints": [
    {"type": "required", "target": "必须发生的事件"},
    {"type": "forbidden", "target": "禁止发生的事件"}
  ],
  "metadata": {
    "chapter": 1,
    "scene_index": 0
  }
}

重要规则：
- execution.units 中的 label 必须为 "action"
- density=dense，生成 3-5 个 units
- state_changes 中的 type 只能是：plot_flag, relationship, inventory, realm, location, hp
- 场景背景：林逸在修仙世界中探索，目标是变强

只输出 JSON，不要有任何额外文本。
"""

def build_segment_prompt(
    contract: PlanningContract,
    segment_units: List[ExecutionUnit],
    current_state: Dict,
    completed_events_summary: str,
    previous_text: str,
    segment_idx: int,
    total_segments: int,
) -> str:
    """构建每个段落的 Writer Prompt（增强版）"""
    lines = []
    lines.append(f"请写一段场景正文（约 300-400 字）。这是第 {segment_idx + 1}/{total_segments} 段。")
    lines.append("")
    lines.append("【场景总目标】")
    lines.append(f"目标：{contract.intent.goal}")
    lines.append(f"冲突：{contract.intent.conflict}")
    lines.append("")

    if previous_text:
        lines.append("【上一段结尾】")
        lines.append(previous_text[-300:])
        lines.append("请自然衔接上一段结尾。")
        lines.append("")

    if completed_events_summary:
        lines.append("【已完成的事件】")
        lines.append(completed_events_summary)
        lines.append("")

    if current_state:
        lines.append("【当前世界状态】")
        state_summary = []
        for key, value in current_state.items():
            if key == "global_flags" and value:
                flags = ", ".join(f"{k}={v}" for k, v in value.items())
                state_summary.append(f"  全局标记: {flags}")
            elif key == "relationships" and value:
                rels = ", ".join(f"{k}:{v}" for k, v in value.items())
                state_summary.append(f"  关系: {rels}")
            elif key == "characters" and value:
                for name, info in value.items():
                    hp = info.get("hp", "?")
                    realm = info.get("realm", "?")
                    loc = info.get("location", "?")
                    state_summary.append(f"  {name}: HP={hp}, 境界={realm}, 位置={loc}")
        if state_summary:
            lines.extend(state_summary)
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

    # 可观测结果提示
    if contract.observables.state_changes:
        lines.append("【必须记录的状态变化】")
        lines.append("请在生成的事件中记录以下状态变化：")
        for change in contract.observables.state_changes:
            if change.type == "plot_flag":
                lines.append(f"- plot_flag: {change.name} = {change.value}")
                lines.append(f"  事件格式: {{'type': 'plot_flag_set', 'flag': '{change.name}', 'value': {change.value}}}")
            elif change.type == "inventory":
                lines.append(f"- inventory: {change.actor} 获得 {change.item}")
                lines.append(f"  事件格式: {{'type': 'item_acquire', 'actor': '{change.actor}', 'item': '{change.item}'}}")
            elif change.type == "relationship":
                lines.append(f"- relationship: {change.from_char} → {change.to_char} 变化 {change.delta}")
                lines.append(f"  事件格式: {{'type': 'relationship_change', 'from_char': '{change.from_char}', 'to_char': '{change.to_char}', 'delta': {change.delta}}}")
            elif change.type == "realm":
                lines.append(f"- realm: {change.actor} 突破到 {change.to_major_realm}{change.to_minor_stage}层")
                lines.append(f"  事件格式: {{'type': 'realm_upgrade', 'actor': '{change.actor}', 'to_major_realm': '{change.to_major_realm}', 'to_minor_stage': {change.to_minor_stage}}}")
        lines.append("")

    lines.append("【输出格式】")
    lines.append('{"scene_text": "...", "events": [{"type": "...", ...}]}')
    lines.append("只输出 JSON，不要有任何额外文本。")

    return "\n".join(lines)

# ============================================================================
# 核心：优化版 Incremental Writer
# ============================================================================

class IncrementalWriter:
    """优化版增量写入器"""

    def __init__(self):
        self.all_text = ""
        self.all_events = []
        self.current_state = {}
        self.completed_units = []
        self.total_units = []

    def split_units(self, units: List[ExecutionUnit], num_segments: int) -> List[List[ExecutionUnit]]:
        """优化分段策略：按逻辑顺序分组，每段至少2个单元，最多3个"""
        if not units:
            return [[] for _ in range(num_segments)]

        total = len(units)
        if num_segments == 1:
            return [units]

        # 如果单元数少于4，只分2段
        if total <= 3:
            num_segments = min(num_segments, 2)

        # 每段至少2个单元
        min_per_segment = 2
        if total < num_segments * min_per_segment:
            num_segments = max(1, total // min_per_segment)

        # 重新分配
        base = total // num_segments
        remainder = total % num_segments

        segments = []
        idx = 0
        for i in range(num_segments):
            count = base + (1 if i < remainder else 0)
            if count < min_per_segment and idx + count < total:
                count = min(min_per_segment, total - idx)
            segments.append(units[idx:idx + count])
            idx += count

        # 如果还有剩余，加到最后一段
        if idx < total:
            if segments:
                segments[-1].extend(units[idx:])
            else:
                segments.append(units[idx:])

        return segments

    def extract_events_summary(self, events: List[Dict]) -> str:
        """从事件中提取摘要"""
        if not events:
            return "无"
        summaries = []
        for evt in events:
            evt_type = evt.get("type", "")
            if evt_type == "plot_flag_set":
                summaries.append(f"设置标记 {evt.get('flag')} = {evt.get('value')}")
            elif evt_type == "item_acquire":
                summaries.append(f"{evt.get('actor')} 获得 {evt.get('item')}")
            elif evt_type == "location_enter":
                summaries.append(f"{evt.get('actor')} 进入 {evt.get('location')}")
            elif evt_type == "realm_upgrade":
                summaries.append(f"{evt.get('actor')} 突破到 {evt.get('to_major_realm')}{evt.get('to_minor_stage')}层")
            elif evt_type == "relationship_change":
                summaries.append(f"{evt.get('from_char')} 与 {evt.get('to_char')} 关系变化 {evt.get('delta')}")
            elif evt_type == "hp_changed":
                summaries.append(f"{evt.get('actor')} HP 变为 {evt.get('new_hp')}")
        return "；".join(summaries) if summaries else "无"

    def apply_events(self, events: List[Dict]) -> Dict:
        """将事件应用到当前状态"""
        state = self.current_state.copy()
        if not state:
            state = {"characters": {"林逸": {"hp": 100, "realm": "炼气", "level": 1, "inventory": []}}, "relationships": {}, "global_flags": {}}

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

    def verify_segment(self, text: str, segment_units: List[ExecutionUnit]) -> bool:
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

    async def write_segment(
        self,
        contract: PlanningContract,
        segment_units: List[ExecutionUnit],
        segment_idx: int,
        total_segments: int,
        is_retry: bool = False,
    ) -> Tuple[str, List[Dict]]:
        """生成一个段落（带重试）"""
        completed_summary = self.extract_events_summary(self.all_events)
        prompt = build_segment_prompt(
            contract=contract,
            segment_units=segment_units,
            current_state=self.current_state,
            completed_events_summary=completed_summary,
            previous_text=self.all_text,
            segment_idx=segment_idx,
            total_segments=total_segments,
        )

        # 如果是重试，增加提示
        if is_retry:
            prompt += "\n\n【上一轮验证失败，请重新生成】"
            prompt += "\n确保完成本段所有执行单元，并输出有效的 JSON。"

        response = await call_llm_with_retry(prompt, max_tokens=2048)
        if not response:
            return "", []

        data = parse_json_response(response)
        if not data:
            return "", []

        text = data.get("scene_text", "")
        events = data.get("events", [])

        return text, events

    async def write_scene(
        self,
        contract: PlanningContract,
        num_segments: int = 2,
    ) -> Tuple[str, List[Dict], Dict]:
        """执行完整的增量写作（优化版）"""
        self.all_text = ""
        self.all_events = []
        self.current_state = {}
        self.completed_units = []

        # 1. 拆分执行单元
        units = contract.execution.units
        self.total_units = units
        segment_units_list = self.split_units(units, num_segments)

        logger.info(f"拆分 {len(units)} 个执行单元为 {len(segment_units_list)} 段")

        # 2. 逐段生成
        for idx, units_in_segment in enumerate(segment_units_list):
            if not units_in_segment:
                continue

            logger.info(f"  段 {idx+1}/{len(segment_units_list)}: {len(units_in_segment)} 个单元")

            # 尝试生成
            text, events = await self.write_segment(
                contract=contract,
                segment_units=units_in_segment,
                segment_idx=idx,
                total_segments=len(segment_units_list),
                is_retry=False,
            )

            # 验证本段是否成功
            if text and self.verify_segment(text, units_in_segment):
                self.all_text += text + "\n\n"
                self.all_events.extend(events)
                self.current_state = self.apply_events(events)
                self.completed_units.extend(units_in_segment)
                logger.info(f"    段 {idx+1} 成功")
            else:
                # 重试一次
                logger.warning(f"    段 {idx+1} 失败，重试...")
                text, events = await self.write_segment(
                    contract=contract,
                    segment_units=units_in_segment,
                    segment_idx=idx,
                    total_segments=len(segment_units_list),
                    is_retry=True,
                )
                if text and self.verify_segment(text, units_in_segment):
                    self.all_text += text + "\n\n"
                    self.all_events.extend(events)
                    self.current_state = self.apply_events(events)
                    self.completed_units.extend(units_in_segment)
                    logger.info(f"    段 {idx+1} 重试成功")
                else:
                    # 降级：单次生成
                    logger.warning(f"    段 {idx+1} 重试失败，降级到单次生成")
                    # 使用完整的 execution.units
                    fallback_prompt = build_segment_prompt(
                        contract=contract,
                        segment_units=contract.execution.units,
                        current_state={},
                        completed_events_summary="",
                        previous_text="",
                        segment_idx=0,
                        total_segments=1,
                    )
                    fallback_response = await call_llm_with_retry(fallback_prompt, max_tokens=4096)
                    fallback_data = parse_json_response(fallback_response)
                    if fallback_data:
                        self.all_text = fallback_data.get("scene_text", "")
                        self.all_events = fallback_data.get("events", [])
                    break

        return self.all_text, self.all_events, self.current_state

# ============================================================================
# Control 验证（与 Phase 1 相同）
# ============================================================================

def validate_contract_units(contract: PlanningContract, text: str) -> Dict[str, Any]:
    if not contract.execution.units:
        return {"completed": 0, "total": 0, "score": 1.0}

    completed = 0
    for unit in contract.execution.units:
        keywords = re.findall(r'[\u4e00-\u9fff]{2,4}', unit.description)
        if not keywords:
            keywords = [unit.description[:6]]
        if any(kw in text for kw in keywords):
            completed += 1

    total = len(contract.execution.units)
    return {"completed": completed, "total": total, "score": completed / total if total > 0 else 1.0}

def validate_contract_constraints(contract: PlanningContract, text: str) -> Dict[str, Any]:
    if not contract.constraints:
        return {"passed": True, "score": 1.0}

    all_passed = True
    for c in contract.constraints:
        keywords = re.findall(r'[\u4e00-\u9fff]{2,4}', c.target)
        if not keywords:
            keywords = [c.target[:6]]

        if c.type == "required":
            if not any(kw in text for kw in keywords):
                all_passed = False
        elif c.type == "forbidden":
            if any(kw in text for kw in keywords):
                all_passed = False

    return {"passed": all_passed, "score": 1.0 if all_passed else 0.0}

def validate_contract_observables(contract: PlanningContract, events: List[Dict]) -> Dict[str, Any]:
    if not contract.observables.state_changes:
        return {"matched": 0, "total": 0, "score": 1.0}

    TYPE_MAP = {
        "plot_flag_set": "plot_flag",
        "flag_set": "plot_flag",
        "relationship_change": "relationship",
        "item_acquire": "inventory",
        "item_lose": "inventory",
        "realm_upgrade": "realm",
        "location_enter": "location",
        "hp_changed": "hp",
    }

    matched = 0
    for change in contract.observables.state_changes:
        found = False
        for evt in events:
            evt_type = evt.get("type", "")
            mapped_type = TYPE_MAP.get(evt_type, evt_type)
            if mapped_type != change.type:
                continue

            if change.type == "plot_flag":
                flag_name = evt.get("name") or evt.get("flag") or evt.get("flag_name")
                if flag_name == change.name and evt.get("value") == change.value:
                    found = True
                    break
            elif change.type == "relationship":
                from_char = evt.get("from_char") or evt.get("from")
                to_char = evt.get("to_char") or evt.get("to")
                if from_char == change.from_char and to_char == change.to_char:
                    delta = evt.get("delta", 0)
                    if abs(delta - change.delta) <= 5:
                        found = True
                        break
            elif change.type == "inventory":
                actor = evt.get("actor") or evt.get("character")
                item = evt.get("item") or evt.get("item_name")
                if actor == change.actor and item == change.item:
                    op = evt.get("operation") or evt.get("action")
                    if op in ["acquire", "add", "get"]:
                        found = True
                        break
                    elif op in ["lose", "remove", "drop"]:
                        found = True
                        break
            elif change.type == "realm":
                actor = evt.get("actor") or evt.get("character")
                if actor == change.actor and evt.get("to_major_realm") == change.to_major_realm:
                    if evt.get("to_minor_stage") == change.to_minor_stage:
                        found = True
                        break
            elif change.type == "location":
                actor = evt.get("actor") or evt.get("character")
                if actor == change.actor and evt.get("location") == change.location:
                    found = True
                    break
            elif change.type == "hp":
                actor = evt.get("actor") or evt.get("character")
                if actor == change.actor and evt.get("new_hp") == change.new_hp:
                    found = True
                    break
        if found:
            matched += 1

    total = len(contract.observables.state_changes)
    score = matched / total if total > 0 else 1.0
    return {"matched": matched, "total": total, "score": score}

# ============================================================================
# 单个场景执行
# ============================================================================

async def run_single_scene(
    group_config: Dict,
    scene_idx: int,
    rep: int,
    writer: IncrementalWriter,
) -> Dict:
    """执行单个场景"""
    execution_type = group_config.get("execution", "single")
    num_segments = group_config.get("segments", 1)

    try:
        # 1. Planner: 生成 Contract
        planner_prompt = build_planner_prompt()
        planner_response = await call_llm_with_retry(planner_prompt, max_tokens=4096)
        contract_data = parse_json_response(planner_response)

        if "planning_contract" in contract_data:
            contract_data = contract_data["planning_contract"]

        # 清理过滤
        if "observables" in contract_data and isinstance(contract_data["observables"], dict):
            if "state_changes" in contract_data["observables"]:
                allowed_types = ["plot_flag", "relationship", "inventory", "realm", "location", "hp"]
                original = contract_data["observables"]["state_changes"]
                if isinstance(original, list):
                    filtered = [sc for sc in original if isinstance(sc, dict) and sc.get("type") in allowed_types]
                    contract_data["observables"]["state_changes"] = filtered
            contract_data["observables"]["story_events"] = []
            contract_data["observables"]["narrative_flags"] = []

        try:
            contract = PlanningContract(**contract_data)
        except Exception as e:
            return {"error": f"Contract validation failed: {e}", "group": group_config["id"]}

        # 2. Writer: 增量或单次
        if execution_type == "single":
            writer_prompt = build_segment_prompt(
                contract=contract,
                segment_units=contract.execution.units,
                current_state={},
                completed_events_summary="",
                previous_text="",
                segment_idx=0,
                total_segments=1,
            )
            response = await call_llm_with_retry(writer_prompt, max_tokens=4096)
            data = parse_json_response(response)
            scene_text = data.get("scene_text", "")
            events = data.get("events", [])
        else:
            scene_text, events, final_state = await writer.write_scene(
                contract=contract,
                num_segments=num_segments,
            )

        # 3. 验证 Control
        surface = validate_contract_units(contract, scene_text)
        constraint = validate_contract_constraints(contract, scene_text)
        outcome = validate_contract_observables(contract, events)

        control_score = (surface["score"] + constraint["score"] + outcome["score"]) / 3

        return {
            "group": group_config["id"],
            "scene_idx": scene_idx,
            "rep": rep,
            "execution": execution_type,
            "segments": num_segments if execution_type != "single" else 1,
            "contract": contract.model_dump(mode='json'),
            "scene_text": scene_text,
            "events": events,
            "control": {
                "surface": surface,
                "constraint": constraint,
                "outcome": outcome,
                "overall": control_score,
            },
            "error": None,
        }
    except Exception as e:
        return {"error": str(e), "group": group_config["id"]}

# ============================================================================
# 主实验流程
# ============================================================================

async def run_experiment(config_path: Path, output_dir: Path):
    config_data = load_config(config_path)
    groups = config_data["groups"]
    scenes_per_group = config_data.get("scenes_per_group", 5)
    repetitions = config_data.get("repetitions", 1)

    writer = IncrementalWriter()
    all_results = []

    for group in groups:
        gid = group["id"]
        logger.info(f"开始组 {gid} (exec={group.get('execution', 'single')})")
        for i in range(scenes_per_group):
            for r in range(repetitions):
                logger.info(f"  场景 {i+1}/{scenes_per_group}, 重复 {r+1}/{repetitions}")
                result = await run_single_scene(group, i, r, writer)
                all_results.append(result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"results_phase2_optimized_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    report = generate_report(all_results, groups)
    report_file = output_dir / f"report_phase2_optimized_{timestamp}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"实验完成: {results_file}")
    logger.info(f"报告已生成: {report_file}")

def generate_report(results: List[Dict], groups: List[Dict]) -> str:
    lines = ["# Phase 2 实验报告（Incremental Writer 优化版）\n"]
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    group_stats = {}
    for g in groups:
        gid = g["id"]
        items = [r for r in results if r.get("group") == gid and not r.get("error")]
        if not items:
            group_stats[gid] = {"count": 0}
            continue

        surface = [it["control"]["surface"]["score"] for it in items]
        constraint = [it["control"]["constraint"]["score"] for it in items]
        outcome = [it["control"]["outcome"]["score"] for it in items]
        overall = [it["control"]["overall"] for it in items]

        import statistics
        group_stats[gid] = {
            "count": len(items),
            "surface_mean": statistics.mean(surface) if surface else 0,
            "constraint_mean": statistics.mean(constraint) if constraint else 0,
            "outcome_mean": statistics.mean(outcome) if outcome else 0,
            "overall_mean": statistics.mean(overall) if overall else 0,
        }

    lines.append("## 各组成绩\n")
    lines.append("| 组 | 执行方式 | 分段数 | Surface | Constraint | Outcome | Overall |")
    lines.append("|---|---|---|---|---|---|---|")

    for g in groups:
        gid = g["id"]
        exec_type = g.get("execution", "single")
        segs = g.get("segments", 1)
        s = group_stats.get(gid, {})
        if s.get("count", 0) == 0:
            lines.append(f"| {gid} | {exec_type} | {segs} | 无数据 | - | - | - |")
        else:
            lines.append(
                f"| {gid} | {exec_type} | {segs} | "
                f"{s['surface_mean']:.3f} | {s['constraint_mean']:.3f} | "
                f"{s['outcome_mean']:.3f} | {s['overall_mean']:.3f} |"
            )

    lines.append("")
    valid_stats = [(k, v) for k, v in group_stats.items() if v.get("count", 0) > 0]
    if valid_stats:
        best_group = max(valid_stats, key=lambda x: x[1].get("overall_mean", 0))
        lines.append(f"## 结论\n")
        lines.append(f"**最佳总体 Control Score**: 组 {best_group[0]} ({best_group[1].get('overall_mean', 0):.3f})")

        # 比较增量 vs 单次
        single_groups = [g for g in groups if g.get("execution") == "single"]
        inc_groups = [g for g in groups if g.get("execution") == "incremental"]

        if single_groups and inc_groups:
            single_stats = [group_stats[g["id"]] for g in single_groups if g["id"] in group_stats and group_stats[g["id"]].get("count", 0) > 0]
            inc_stats = [group_stats[g["id"]] for g in inc_groups if g["id"] in group_stats and group_stats[g["id"]].get("count", 0) > 0]

            if single_stats and inc_stats:
                single_mean = sum(s["overall_mean"] for s in single_stats) / len(single_stats)
                inc_mean = sum(s["overall_mean"] for s in inc_stats) / len(inc_stats)
                lines.append("")
                lines.append(f"### 对比\n")
                lines.append(f"- 单次执行平均 Overall: {single_mean:.3f}")
                lines.append(f"- 增量执行平均 Overall: {inc_mean:.3f}")
                if inc_mean > single_mean:
                    lines.append(f"- **增量执行提升: +{(inc_mean - single_mean):.3f}**")
                else:
                    lines.append(f"- 增量执行未显著提升")

    return "\n".join(lines)

# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/phase2/config.yaml")
    parser.add_argument("--output", default="experiments/phase2/reports")
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(run_experiment(config_path, output_dir))