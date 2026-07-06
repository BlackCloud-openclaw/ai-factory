#!/usr/bin/env python3
"""
Phase 1 实验运行器（完全独立版）
不导入任何 src.agents / src.execution / src.orchestrator / src.writing 模块
直接调用 LLM API，使用硬编码配置
"""

import asyncio
import json
import yaml
import logging
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import httpx
from openai import AsyncOpenAI

# 从本地导入 Planning Contract（完全独立，无循环依赖）
from experiments.phase1.planning_contract_local import (
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
logger = logging.getLogger("phase1_standalone")

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

# ============================================================================
# LLM 调用
# ============================================================================

async def call_llm(prompt: str, max_tokens: int = 4096) -> str:
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
    return response.choices[0].message.content or ""

# ============================================================================
# Prompt 构建
# ============================================================================

def build_planner_prompt(representation: str, density: str) -> str:
    return f"""
你是一位专业的小说规划师。请根据以下要求生成一个场景的 Planning Contract。

表示方式: {representation}
信息密度: {density}

输出格式必须是符合以下 Schema 的 JSON：

{{
  "version": "1.0",
  "scene_id": "scene_001",
  "intent": {{
    "goal": "场景目标（一句话）",
    "conflict": "核心冲突（一句话）",
    "expected_outcome": "预期结果（一句话）"
  }},
  "execution": {{
    "units": [
      {{"id": "U1", "label": "{representation}", "description": "执行步骤描述", "attributes": {{}}}}
    ]
  }},
  "observables": {{
    "state_changes": [
      {{"type": "plot_flag", "name": "flag_name", "value": true}}
    ],
    "story_events": [],
    "narrative_flags": []
  }},
  "constraints": [
    {{"type": "required", "target": "必须发生的事件"}},
    {{"type": "forbidden", "target": "禁止发生的事件"}}
  ],
  "metadata": {{
    "chapter": 1,
    "scene_index": 0
  }}
}}

重要规则：
- 如果表示方式是 "action"，execution.units 中的 label 必须为 "action"
- 如果表示方式是 "beat"，label 必须为 "beat"
- 如果表示方式是 "intent"，label 必须为 "intent"
- 如果表示方式是 "constraint"，label 必须为 "constraint"
- 如果表示方式是 "summary"，label 必须为 "intent"
- {density} 密度：如果 density=sparse，只生成 1-2 个 units；如果 density=dense，生成 3-5 个 units
- 场景背景：林逸在修仙世界中探索，目标是变强
- **state_changes 中的 type 只能是以下之一：plot_flag, relationship, inventory, realm, location, hp**
- **state_changes 必须是一个对象数组，每个对象包含 type 和相应的字段**

只输出 JSON，不要有任何额外文本。
"""

def build_writer_prompt(contract: PlanningContract) -> str:
    lines = []
    lines.append("请根据以下 Planning Contract 写一个场景正文（约 500-800 字）。")
    lines.append("")
    lines.append("【📋 规划契约】")
    lines.append(f"场景目标：{contract.intent.goal}")
    lines.append(f"核心冲突：{contract.intent.conflict}")
    lines.append(f"预期结果：{contract.intent.expected_outcome}")
    lines.append("")
    
    if contract.execution.units:
        lines.append("【必须完成的执行单元】")
        for unit in contract.execution.units:
            lines.append(f"- {unit.label}: {unit.description}")
        lines.append("")
    
    if contract.constraints:
        lines.append("【硬性约束】")
        for c in contract.constraints:
            if c.type == "required":
                lines.append(f"  ✅ 必须发生：{c.target}")
            elif c.type == "forbidden":
                lines.append(f"  ❌ 禁止发生：{c.target}")
        lines.append("")
    
    # ========== 新增：Observable Outcomes（关键！）==========
    if contract.observables.state_changes:
        lines.append("【📊 必须记录的状态变化（请生成对应的事件）】")
        lines.append("场景结束后，世界状态必须发生以下变化。你必须在 `events` 数组中用以下格式记录：")
        lines.append("")
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
            elif change.type == "location":
                lines.append(f"- location: {change.actor} 进入 {change.location}")
                lines.append(f"  事件格式: {{'type': 'location_enter', 'actor': '{change.actor}', 'location': '{change.location}'}}")
        lines.append("")
        lines.append("⚠️ 请确保上述状态变化在正文中自然发生，并在 `events` 数组中用指定格式记录！")
        lines.append("")
    # ==========================================================
    
    lines.append("【输出格式】")
    lines.append("输出 JSON 格式：")
    lines.append('{"scene_text": "...", "events": [{"type": "...", ...}]}')
    lines.append("")
    lines.append("只输出 JSON，不要有任何额外文本。")
    
    return "\n".join(lines)

# ============================================================================
# 解析辅助（修复）
# ============================================================================

def parse_json_response(text: str) -> Dict:
    """提取 JSON，确保返回字典"""
    if not text or not isinstance(text, str):
        return {}
    # 清理文本
    text = text.strip()
    # 尝试提取 JSON 对象
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            # 尝试修复常见错误
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

# ============================================================================
# Control 验证
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
    """验证可观测结果是否发生"""
    if not contract.observables.state_changes:
        return {"matched": 0, "total": 0, "score": 1.0}

    # 类型映射
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

            # 根据类型匹配
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
) -> Dict:
    representation = group_config.get("representation", "action")
    density = group_config.get("density", "sparse")

    try:
        planner_prompt = build_planner_prompt(representation, density)
        planner_response = await call_llm(planner_prompt, max_tokens=4096)
        contract_data = parse_json_response(planner_response)

        if "planning_contract" in contract_data:
            contract_data = contract_data["planning_contract"]

        # ========== 清理和过滤 ==========
        if not isinstance(contract_data, dict):
            return {"error": f"Invalid contract data type: {type(contract_data)}", "group": group_config["id"]}

        if "observables" in contract_data and isinstance(contract_data["observables"], dict):
            # 过滤 state_changes
            if "state_changes" in contract_data["observables"]:
                allowed_types = ["plot_flag", "relationship", "inventory", "realm", "location", "hp"]
                original = contract_data["observables"]["state_changes"]
                if isinstance(original, list):
                    filtered = []
                    for sc in original:
                        if isinstance(sc, dict) and sc.get("type") in allowed_types:
                            filtered.append(sc)
                    contract_data["observables"]["state_changes"] = filtered
                else:
                    contract_data["observables"]["state_changes"] = []
            # 清空 story_events 和 narrative_flags
            contract_data["observables"]["story_events"] = []
            contract_data["observables"]["narrative_flags"] = []
        # =====================================

        try:
            contract = PlanningContract(**contract_data)
        except Exception as e:
            return {"error": f"Contract validation failed: {e}", "group": group_config["id"]}

        writer_prompt = build_writer_prompt(contract)
        writer_response = await call_llm(writer_prompt, max_tokens=4096)
        writer_data = parse_json_response(writer_response)
        scene_text = writer_data.get("scene_text", "")
        events = writer_data.get("events", [])

        surface = validate_contract_units(contract, scene_text)
        constraint = validate_contract_constraints(contract, scene_text)
        outcome = validate_contract_observables(contract, events)

        control_score = (surface["score"] + constraint["score"] + outcome["score"]) / 3

        return {
            "group": group_config["id"],
            "scene_idx": scene_idx,
            "rep": rep,
            "representation": representation,
            "density": density,
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
    scenes_per_group = config_data.get("scenes_per_group", 2)
    repetitions = config_data.get("repetitions", 1)

    all_results = []

    for group in groups:
        gid = group["id"]
        logger.info(f"开始组 {gid} (rep={group['representation']}, density={group['density']})")
        for i in range(scenes_per_group):
            for r in range(repetitions):
                logger.info(f"  场景 {i+1}/{scenes_per_group}, 重复 {r+1}/{repetitions}")
                result = await run_single_scene(group, i, r)
                all_results.append(result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"results_contract_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    report = generate_report(all_results, groups)
    report_file = output_dir / f"report_contract_{timestamp}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"实验完成: {results_file}")
    logger.info(f"报告已生成: {report_file}")

def generate_report(results: List[Dict], groups: List[Dict]) -> str:
    lines = ["# Phase 1 实验报告（基于 Planning Contract - 完全独立版）\n"]
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
    lines.append("| 组 | 表示 | 密度 | Surface | Constraint | Outcome | Overall |")
    lines.append("|---|---|---|---|---|---|---|")

    for g in groups:
        gid = g["id"]
        s = group_stats.get(gid, {})
        if s.get("count", 0) == 0:
            lines.append(f"| {gid} | {g['representation']} | {g['density']} | 无数据 | - | - | - |")
        else:
            lines.append(
                f"| {gid} | {g['representation']} | {g['density']} | "
                f"{s['surface_mean']:.3f} | {s['constraint_mean']:.3f} | "
                f"{s['outcome_mean']:.3f} | {s['overall_mean']:.3f} |"
            )

    lines.append("")
    valid_stats = [(k, v) for k, v in group_stats.items() if v.get("count", 0) > 0]
    if valid_stats:
        best_group = max(valid_stats, key=lambda x: x[1].get("overall_mean", 0))
        lines.append(f"## 结论\n")
        lines.append(f"**最佳总体 Control Score**: 组 {best_group[0]} ({best_group[1].get('overall_mean', 0):.3f})")

    return "\n".join(lines)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/phase1/config.yaml")
    parser.add_argument("--output", default="experiments/phase1/reports")
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(run_experiment(config_path, output_dir))