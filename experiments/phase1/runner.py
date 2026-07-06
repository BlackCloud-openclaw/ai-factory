# experiments/phase1/runner.py
import asyncio
import json
import yaml
import logging
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 只导入必要的非循环模块（db, metrics 不涉及 writing）
from src.db import init_db_pool, close_db_pool
from experiments.phase1.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phase1_runner")

FIXED_OUTLINE = {
    "title": "修仙路",
    "volumes": [{
        "volume_num": 1,
        "chapters": [{
            "chapter_num": 1,
            "must_events": ["林逸进入青云宗外门", "遇见大师兄", "发现玉佩异常"],
        }]
    }]
}

def get_initial_world() -> Dict:
    """返回简单的状态字典，不依赖 WorldState"""
    return {
        "characters": {
            "林逸": {
                "name": "林逸",
                "realm": "炼气",
                "realm_level": 1,
                "hp": 100,
                "mp": 100,
                "inventory": [],
                "relationships": {},
                "location": "",
                "flags": {},
            }
        },
        "items": {},
        "relationships": {},
        "map": {"current": "", "locations": {}, "unlocked_regions": []},
        "global_flags": {},
        "recent_event_ids": [],
        "phase_transitions": [],
        "attractor_field": {},
        "constraints": [],
        "version": "2.0",
        "revision": 0,
    }

def apply_events_simple(state: Dict, events: List[Dict]) -> Dict:
    """手动应用事件到状态字典（简化版）"""
    import copy
    new_state = copy.deepcopy(state)
    chars = new_state.get("characters", {})
    for evt in events:
        evt_type = evt.get("type")
        if evt_type == "realm_upgrade":
            actor = evt.get("actor")
            to_major = evt.get("to_major_realm")
            to_stage = evt.get("to_minor_stage")
            if actor in chars:
                chars[actor]["realm"] = to_major
                chars[actor]["realm_level"] = to_stage
        elif evt_type == "item_acquire":
            actor = evt.get("actor")
            item = evt.get("item")
            if actor in chars:
                if "inventory" not in chars[actor]:
                    chars[actor]["inventory"] = []
                chars[actor]["inventory"].append(item)
        elif evt_type == "relationship_change":
            from_char = evt.get("from_char")
            to_char = evt.get("to_char")
            delta = evt.get("delta")
            key = f"{from_char}|{to_char}"
            rels = new_state.get("relationships", {})
            old = rels.get(key, 0)
            rels[key] = old + delta
            new_state["relationships"] = rels
        elif evt_type == "hp_changed":
            actor = evt.get("actor")
            new_hp = evt.get("new_hp")
            if actor in chars:
                chars[actor]["hp"] = new_hp
        elif evt_type == "plot_flag_set":
            flag = evt.get("flag")
            value = evt.get("value", True)
            flags = new_state.get("global_flags", {})
            flags[flag] = value
            new_state["global_flags"] = flags
    return new_state

def load_config(path: Path) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

async def run_single_scene(
    group_id: str,
    scene_idx: int,
    rep: int,
    config: Dict,
    outline: Dict,
    initial_state: Dict,
) -> Dict:
    # 延迟导入 Agent（这些会触发 writing 导入，但已经在函数内，不会影响顶层）
    from src.agents.planner import PlannerAgent
    from src.agents.writer import WritingAgent
    from src.agents.validator import ValidatorAgent
    # 导入 PlanningDirective（同样会触发 writing/__init__，但可以忍受，因为已经在函数内）
    from src.writing.planning_directive import PlanningDirective

    planning_style = config["representation"]
    density = config["density"]
    execution = config["execution"]

    state = SimpleNamespace(
        novel_id=f"exp_{group_id}_s{scene_idx}_r{rep}",
        task_type="scene_plan",
        metadata={
            "planning_style": planning_style,
            "information_density": density,
            "execution_strategy": execution,
        },
        outline=outline,
        current_state=initial_state,
        current_volume=1,
        current_chapter=1,
        user_input="生成一个场景",
        writing_constraints=None,
        narrative_blueprint=None,
        knowledge_deltas=None,
        character_intent=None,
        drama_structure=None,
        current_scene_index=0,
        voice_memory=None,
        compressed_state=None,
        scene_plan=None,
        scene_plan_list=[],
        total_scenes_in_chapter=0,
        phase=None,
        validation_result=None,
        step_count=0,
        retry_count=0,
        max_retries_per_subtask=3,
        needs_retry=False,
        error=None,
        scene_text="",
        final_answer="",
    )

    # 1. Planner
    planner = PlannerAgent()
    plan_result = await planner.run(state)
    if plan_result.get("error"):
        return {"error": plan_result["error"], "group": group_id, "scene_idx": scene_idx, "rep": rep}

    directive_data = state.metadata.get("planning_directive")
    if not directive_data:
        return {"error": "No planning_directive", "group": group_id, "scene_idx": scene_idx, "rep": rep}
    directive = PlanningDirective(**directive_data)
    scene_plan = plan_result.get("scene_plan", {}).get("scenes", [{}])[0]

    # 2. Writer
    writer = WritingAgent()
    writer_result = await writer.run(state)
    if writer_result.get("error"):
        return {"error": writer_result["error"], "group": group_id, "scene_idx": scene_idx, "rep": rep}

    raw_output = writer_result.get("scene_text", "")
    json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            text = data.get("scene_text", "")
            events_data = data.get("events", [])
        except:
            text = raw_output
            events_data = []
    else:
        text = raw_output
        events_data = []

    # 3. 应用事件得到最终状态（使用简化函数）
    final_state = apply_events_simple(initial_state, events_data)

    # 4. Validator
    val_state = SimpleNamespace(
        scene_text=raw_output,
        scene_plan=scene_plan,
        outline=outline,
        current_state=initial_state,
        novel_id=state.novel_id,
        validation_mode="novel",
        metadata={},
        current_volume=1,
        current_chapter=1,
        current_scene_index=0,
        writing_constraints=None,
        narrative_blueprint=None,
        knowledge_deltas=None,
        character_intent=None,
        drama_structure=None,
        voice_memory=None,
        compressed_state=None,
    )
    validator = ValidatorAgent()
    val_result = await validator.run(val_state)
    validation_result = val_result.get("validation_result", {})

    # 5. 计算指标（需要 directive, text, initial_state, final_state, validation_result）
    metrics = compute_all_metrics(
        directive=directive,
        text=text,
        initial_state=initial_state,
        final_state=final_state,
        validation_result=validation_result,
        expected_projection=None,
        actual_projection=None,
    )

    return {
        "group": group_id,
        "scene_idx": scene_idx,
        "rep": rep,
        "directive": directive.dict(),
        "text": text,
        "events": events_data,
        "final_state": final_state,
        "validation_passed": validation_result.get("passed", False),
        "metrics": metrics,
        "error": None,
    }

async def run_experiment(config_path: Path, output_dir: Path):
    config_data = load_config(config_path)
    groups = config_data["groups"]
    scenes_per_group = config_data.get("scenes_per_group", 10)
    repetitions = config_data.get("repetitions", 2)

    await init_db_pool()
    outline = FIXED_OUTLINE
    initial_state = get_initial_world()
    all_results = []

    for group_config in groups:
        group_id = group_config["id"]
        logger.info(f"开始实验组 {group_id}")
        for scene_idx in range(scenes_per_group):
            for rep in range(repetitions):
                logger.info(f"  场景 {scene_idx+1}/{scenes_per_group}, 重复 {rep+1}/{repetitions}")
                result = await run_single_scene(
                    group_id, scene_idx, rep, group_config, outline, initial_state
                )
                all_results.append(result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"results_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 生成报告
    report_lines = ["# Phase 1 实验报告\n"]
    group_map = {}
    for r in all_results:
        if r.get("error"):
            continue
        gid = r["group"]
        group_map.setdefault(gid, []).append(r)

    import statistics
    for g in groups:
        gid = g["id"]
        items = group_map.get(gid, [])
        if not items:
            report_lines.append(f"## 组 {gid} （无数据）")
            continue
        surface = [it["metrics"]["surface_compliance"] for it in items]
        deep = [it["metrics"]["deep_compliance"] for it in items]
        pred = [it["metrics"]["predictability"] for it in items]
        rewrite = [it["metrics"]["rewrite_rate"] for it in items]
        report_lines.append(f"## 组 {gid} (rep={g['representation']}, density={g['density']}, exec={g['execution']})")
        report_lines.append(f"- 平均表层遵循度: {statistics.mean(surface):.3f}")
        report_lines.append(f"- 平均深层遵循度: {statistics.mean(deep):.3f}")
        report_lines.append(f"- 平均可预测性: {statistics.mean(pred):.3f}")
        report_lines.append(f"- 平均重写率: {statistics.mean(rewrite):.3f}\n")

    report_file = output_dir / f"report_{timestamp}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"实验完成，结果：{results_file}，报告：{report_file}")
    await close_db_pool()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/phase1/config.yaml")
    parser.add_argument("--output", default="experiments/phase1/reports")
    args = parser.parse_args()
    asyncio.run(run_experiment(Path(args.config), Path(args.output)))