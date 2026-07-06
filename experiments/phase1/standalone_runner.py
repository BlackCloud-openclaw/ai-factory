#!/usr/bin/env python3
# experiments/phase1/standalone_runner.py
import asyncio
import json
import yaml
import logging
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import httpx
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 只导入配置和基础工具，不导入任何可能循环的模块
from src.config import config
from src.db import init_db_pool, close_db_pool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phase1_standalone")

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

def load_config(path: Path) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_initial_world() -> Dict:
    return {"characters": {"林逸": {"realm": "炼气", "level": 1}}, "items": {}, "relationships": {}, "global_flags": {}}

# 规划表示枚举（简化）
PLANNING_STYLES = ["summary", "beat", "action", "intent", "constraint"]

def build_planner_prompt(style: str, density: str) -> str:
    if style == "summary":
        return """生成一个场景计划，只提供场景概述。输出JSON格式：
{"scenes": [{"goal": "...", "conflict": "...", "outcome": "...", "characters": ["林逸"], "must_events": []}]}"""
    elif style == "beat":
        return """生成一个场景计划，拆解为节拍列表。输出JSON格式：
{"scenes": [{"goal": "...", "conflict": "...", "outcome": "...", "characters": ["林逸"], "must_events": ["节拍1", "节拍2"]}]}"""
    elif style == "action":
        return """生成一个场景计划，明确具体动作。输出JSON格式：
{"scenes": [{"goal": "...", "conflict": "...", "outcome": "...", "characters": ["林逸"], "must_events": ["动作1", "动作2"]}]}"""
    elif style == "intent":
        return """生成一个场景计划，强调写作意图。输出JSON格式：
{"scenes": [{"goal": "...", "conflict": "...", "outcome": "...", "characters": ["林逸"], "must_events": []}]}"""
    elif style == "constraint":
        return """生成一个场景计划，只提供约束（必须/禁止）。输出JSON格式：
{"scenes": [{"goal": "...", "conflict": "...", "outcome": "...", "characters": ["林逸"], "must_events": [], "forbidden_events": []}]}"""
    else:
        return """生成一个标准场景计划。输出JSON格式：
{"scenes": [{"goal": "...", "conflict": "...", "outcome": "...", "characters": ["林逸"], "must_events": []}]}"""

def build_writer_prompt(scene_plan: Dict, style: str) -> str:
    goal = scene_plan.get("goal", "推进剧情")
    conflict = scene_plan.get("conflict", "面临困难")
    must_events = scene_plan.get("must_events", [])
    
    prompt = f"""写一个小说场景，约800字，必须包含以下要素：
目标：{goal}
冲突：{conflict}
"""
    if style == "summary":
        prompt += f"场景概述：{goal}。面临{conflict}。请自然地展开。"
    elif style == "beat":
        prompt += "必须按顺序完成这些节拍：\n" + "\n".join([f"- {b}" for b in must_events]) if must_events else ""
    elif style == "action":
        prompt += "必须执行这些动作：\n" + "\n".join([f"- {a}" for a in must_events]) if must_events else ""
    elif style == "intent":
        prompt += f"写作意图：{goal}。成功条件：达成{goal}。请通过情节推进达成此意图。"
    elif style == "constraint":
        if must_events:
            prompt += "必须发生：" + "、".join(must_events)
        forbidden = scene_plan.get("forbidden_events", [])
        if forbidden:
            prompt += "\n禁止发生：" + "、".join(forbidden)
    prompt += "\n输出JSON格式：{\"scene_text\": \"...\", \"events\": [{\"type\": \"...\", ...}]}"
    return prompt

async def call_llm(prompt: str, max_tokens: int = 4000) -> str:
    """直接调用本地 llama.cpp 服务（写作模型）"""
    client = AsyncOpenAI(
        api_key="not-needed",
        base_url="http://localhost:8082/v1",  # 写作服务
        timeout=httpx.Timeout(600.0, connect=30.0)
    )
    response = await client.chat.completions.create(
        model="Qwen3-32B-Q5_K_M",  # 或者你的模型名
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""

def parse_json(text: str) -> Dict:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            return {}
    return {}

def compute_surface_compliance(text: str, scene_plan: Dict, style: str) -> float:
    if not text:
        return 0.0
    # 提取关键词
    if style == "summary":
        keywords = re.findall(r'[\u4e00-\u9fff]{2,4}', scene_plan.get("goal", "") + scene_plan.get("conflict", ""))
    elif style in ["beat", "action", "constraint"]:
        events = scene_plan.get("must_events", [])
        keywords = []
        for evt in events:
            keywords.extend(re.findall(r'[\u4e00-\u9fff]{2,4}', evt))
    else:  # intent
        keywords = re.findall(r'[\u4e00-\u9fff]{2,4}', scene_plan.get("goal", ""))
    keywords = list(set(keywords))
    if not keywords:
        return 1.0
    text_clean = re.sub(r'[，。！？；：""“”‘’\n\t]', '', text)
    matched = sum(1 for kw in keywords if kw in text_clean)
    return matched / len(keywords)

def compute_deep_compliance(scene_plan: Dict, final_state: Dict) -> float:
    expected = scene_plan.get("state_delta", {})
    if not expected:
        return 1.0
    # 简化：检查境界
    exp_realm = expected.get("realm")
    if exp_realm:
        final_realm = final_state.get("characters", {}).get("林逸", {}).get("realm")
        return 1.0 if exp_realm == final_realm else 0.0
    return 0.5

async def run_single_scene(group_config: Dict, scene_idx: int, rep: int) -> Dict:
    style = group_config["representation"]
    density = group_config["density"]
    
    # 1. Planner
    planner_prompt = build_planner_prompt(style, density)
    planner_response = await call_llm(planner_prompt, max_tokens=4000)
    plan_data = parse_json(planner_response)
    scenes = plan_data.get("scenes", [])
    if not scenes:
        return {"error": "No scenes generated", "group": group_config["id"], "scene_idx": scene_idx, "rep": rep}
    scene_plan = scenes[0]
    
    # 2. Writer
    writer_prompt = build_writer_prompt(scene_plan, style)
    writer_response = await call_llm(writer_prompt, max_tokens=4000)
    data = parse_json(writer_response)
    scene_text = data.get("scene_text", "")
    events = data.get("events", [])
    
    # 3. 模拟状态变化（简化）
    initial = get_initial_world()
    final = initial.copy()
    for evt in events:
        if evt.get("type") == "realm_upgrade":
            final["characters"]["林逸"]["realm"] = evt.get("to_major_realm", "炼气")
    
    surface = compute_surface_compliance(scene_text, scene_plan, style)
    deep = compute_deep_compliance(scene_plan, final)
    
    return {
        "group": group_config["id"],
        "scene_idx": scene_idx,
        "rep": rep,
        "scene_text": scene_text,
        "events": events,
        "metrics": {"surface_compliance": surface, "deep_compliance": deep},
        "error": None
    }

async def run_experiment(config_path: Path, output_dir: Path):
    config_data = load_config(config_path)
    groups = config_data["groups"]
    scenes_per_group = config_data.get("scenes_per_group", 2)
    repetitions = config_data.get("repetitions", 1)
    
    all_results = []
    for group in groups:
        gid = group["id"]
        logger.info(f"开始组 {gid}")
        for i in range(scenes_per_group):
            for r in range(repetitions):
                logger.info(f"  场景 {i+1}/{scenes_per_group}, 重复 {r+1}/{repetitions}")
                result = await run_single_scene(group, i, r)
                all_results.append(result)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"results_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    report_lines = ["# Phase 1 实验报告（独立模式）\n"]
    for g in groups:
        gid = g["id"]
        items = [r for r in all_results if r.get("group") == gid and not r.get("error")]
        if not items:
            report_lines.append(f"## 组 {gid} 无数据")
            continue
        surface = [it["metrics"]["surface_compliance"] for it in items]
        deep = [it["metrics"]["deep_compliance"] for it in items]
        import statistics
        report_lines.append(f"## 组 {gid} (rep={g['representation']}, density={g['density']})")
        report_lines.append(f"- 平均表层遵循度: {statistics.mean(surface):.3f}")
        report_lines.append(f"- 平均深层遵循度: {statistics.mean(deep):.3f}\n")
    
    report_file = output_dir / f"report_{timestamp}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"实验完成，结果：{results_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/phase1/config.yaml")
    parser.add_argument("--output", default="experiments/phase1/reports")
    args = parser.parse_args()
    asyncio.run(run_experiment(Path(args.config), Path(args.output)))