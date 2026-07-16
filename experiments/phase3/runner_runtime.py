#!/usr/bin/env python3
"""
Phase 3 实验运行器 - Narrative Runtime 测试

测试 Runtime 的 2/3/4 段及自适应分段策略，与单次执行对比。
"""

# experiments/phase3/runner_runtime.py
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

# 从本地导入 Runtime 和 Contract（避免循环导入）
from experiments.phase3.runtime_local import NarrativeRuntime
from experiments.phase1.planning_contract_local import PlanningContract

# 验证函数从 Phase 2 的本地 runner 导入（也是独立的）
from experiments.phase2.runner_incremental_optimized import (
    validate_contract_units,
    validate_contract_constraints,
    validate_contract_observables,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phase3_runtime")

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

async def call_llm(prompt: str, max_tokens: int = 4096) -> str:
    """通用 LLM 调用（无代理）"""
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
# Planner Prompt（固定 Action + Dense）
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

# ============================================================================
# 单个场景执行
# ============================================================================

async def run_single_scene(
    group_config: Dict,
    scene_idx: int,
    rep: int,
) -> Dict:
    """执行单个场景（使用 Runtime 或单次）"""
    use_runtime = group_config.get("runtime", True)
    segments_mode = group_config.get("segments", "auto")  # auto, 2, 3, 4

    try:
        # 1. Planner: 生成 Contract
        planner_prompt = build_planner_prompt()
        planner_response = await call_llm(planner_prompt, max_tokens=4096)
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

        # 2. 执行：单次或 Runtime
        if use_runtime:
            if segments_mode == "auto":
                # 传入 None，让 Runtime 自动决定
                runtime = NarrativeRuntime(default_segments=2)
                runtime_result = await runtime.execute(contract, segments_hint=None)
            else:
                runtime = NarrativeRuntime(default_segments=int(segments_mode))
                runtime_result = await runtime.execute(contract, segments_hint=int(segments_mode))

            scene_text = runtime_result.full_text
            events = runtime_result.all_events
            final_state = runtime_result.final_state

            # 记录 Runtime 元数据
            runtime_meta = {
                "total_segments": runtime_result.total_segments,
                "successful_segments": runtime_result.successful_segments,
                "execution_time": runtime_result.execution_time,
            }
        else:
            # 单次执行（与 Phase 2 相同）
            logger.info(f"  📝 单次执行")
            # 复用 Phase 2 的单次 prompt 构建
            from experiments.phase2.runner_incremental_optimized import build_segment_prompt
            writer_prompt = build_segment_prompt(
                contract=contract,
                segment_units=contract.execution.units,
                current_state={},
                completed_events_summary="",
                previous_text="",
                segment_idx=0,
                total_segments=1,
            )
            response = await call_llm(writer_prompt, max_tokens=4096)
            data = parse_json_response(response)
            scene_text = data.get("scene_text", "")
            events = data.get("events", [])
            final_state = {}
            runtime_meta = {}

        # 3. 验证 Control
        surface = validate_contract_units(contract, scene_text)
        constraint = validate_contract_constraints(contract, scene_text)
        outcome = validate_contract_observables(contract, events)

        control_score = (surface["score"] + constraint["score"] + outcome["score"]) / 3

        return {
            "group": group_config["id"],
            "scene_idx": scene_idx,
            "rep": rep,
            "runtime": use_runtime,
            "segments": runtime_meta.get("total_segments", 1) if use_runtime else 1,
            "successful_segments": runtime_meta.get("successful_segments", 0),
            "execution_time": runtime_meta.get("execution_time", 0),
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

    all_results = []

    for group in groups:
        gid = group["id"]
        logger.info(f"开始组 {gid} (runtime={group.get('runtime', False)}, segments={group.get('segments', 'auto')})")
        for i in range(scenes_per_group):
            for r in range(repetitions):
                logger.info(f"  场景 {i+1}/{scenes_per_group}, 重复 {r+1}/{repetitions}")
                result = await run_single_scene(group, i, r)
                all_results.append(result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"results_phase3_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    report = generate_report(all_results, groups)
    report_file = output_dir / f"report_phase3_{timestamp}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"实验完成: {results_file}")
    logger.info(f"报告已生成: {report_file}")

def generate_report(results: List[Dict], groups: List[Dict]) -> str:
    lines = ["# Phase 3 实验报告（Narrative Runtime）\n"]
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
        exec_time = [it.get("execution_time", 0) for it in items]
        success_seg = [it.get("successful_segments", 0) for it in items]
        total_seg = [it.get("segments", 1) for it in items]

        import statistics
        group_stats[gid] = {
            "count": len(items),
            "surface_mean": statistics.mean(surface) if surface else 0,
            "constraint_mean": statistics.mean(constraint) if constraint else 0,
            "outcome_mean": statistics.mean(outcome) if outcome else 0,
            "overall_mean": statistics.mean(overall) if overall else 0,
            "avg_exec_time": statistics.mean(exec_time) if exec_time else 0,
            "avg_success_seg": statistics.mean(success_seg) if success_seg else 0,
            "avg_total_seg": statistics.mean(total_seg) if total_seg else 0,
        }

    lines.append("## 各组成绩\n")
    lines.append("| 组 | Runtime | 分段模式 | Surface | Constraint | Outcome | Overall | 耗时(s) | 成功率 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for g in groups:
        gid = g["id"]
        runtime = "✅" if g.get("runtime", False) else "❌"
        seg_mode = str(g.get("segments", 1))
        s = group_stats.get(gid, {})
        if s.get("count", 0) == 0:
            lines.append(f"| {gid} | {runtime} | {seg_mode} | 无数据 | - | - | - | - | - |")
        else:
            success_rate = s["avg_success_seg"] / s["avg_total_seg"] if s["avg_total_seg"] > 0 else 0
            lines.append(
                f"| {gid} | {runtime} | {seg_mode} | "
                f"{s['surface_mean']:.3f} | {s['constraint_mean']:.3f} | "
                f"{s['outcome_mean']:.3f} | {s['overall_mean']:.3f} | "
                f"{s['avg_exec_time']:.2f} | {success_rate:.0%} |"
            )

    lines.append("")
    valid_stats = [(k, v) for k, v in group_stats.items() if v.get("count", 0) > 0]
    if valid_stats:
        best_group = max(valid_stats, key=lambda x: x[1].get("overall_mean", 0))
        lines.append(f"## 结论\n")
        lines.append(f"**最佳总体 Control Score**: 组 {best_group[0]} ({best_group[1].get('overall_mean', 0):.3f})")

        # 对比 Runtime vs 单次
        runtime_groups = [g for g in groups if g.get("runtime", False)]
        single_groups = [g for g in groups if not g.get("runtime", False)]
        if runtime_groups and single_groups:
            runtime_stats = [group_stats[g["id"]] for g in runtime_groups if g["id"] in group_stats and group_stats[g["id"]].get("count", 0) > 0]
            single_stats = [group_stats[g["id"]] for g in single_groups if g["id"] in group_stats and group_stats[g["id"]].get("count", 0) > 0]
            if runtime_stats and single_stats:
                runtime_mean = sum(s["overall_mean"] for s in runtime_stats) / len(runtime_stats)
                single_mean = sum(s["overall_mean"] for s in single_stats) / len(single_stats)
                lines.append("")
                lines.append(f"### 对比\n")
                lines.append(f"- Runtime 平均 Overall: {runtime_mean:.3f}")
                lines.append(f"- 单次执行平均 Overall: {single_mean:.3f}")
                if runtime_mean > single_mean:
                    lines.append(f"- **Runtime 提升: +{(runtime_mean - single_mean):.3f}**")
                else:
                    lines.append(f"- Runtime 未显著提升")

    return "\n".join(lines)

# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/phase3/config.yaml")
    parser.add_argument("--output", default="experiments/phase3/reports")
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(run_experiment(config_path, output_dir))