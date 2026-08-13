#!/usr/bin/env python3
"""
自动标注候选场景（配额控制 + 人工确认）
"""

import json
import yaml
from pathlib import Path
from collections import defaultdict

CANDIDATES_DIR = Path("experiments/phase12/corpus/candidates")
OUTPUT_DIR = Path("experiments/phase12/corpus/v1.0")

TARGET_COUNTS = {
    "scene_transition": 5,
    "character_state": 5,
    "dialogue_quality": 5,
    "planning_execution": 5,
    "runtime_state": 5,
}

# 类别关键词
KEYWORDS = {
    "planning_execution": ["计划", "目标", "执行", "任务", "完成", "必须", "契约", "规划"],
    "runtime_state": ["状态", "快照", "事件", "记录", "更新", "日志", "数据"],
    "dialogue_quality": ["对话", "说", "问", "答", "道", "喊", "叫", "交谈"],
    "character_state": ["境界", "突破", "金丹", "元婴", "修为", "灵力", "血脉", "伤势"],
    "scene_transition": ["时间", "突然", "第二天", "转向", "切换", "转移", "进入"],
}

RUNTIME_ARTIFACTS = ["runtime_metrics", "snapshot_before", "snapshot_after", "events"]


def classify_candidate(data):
    text = data.get("scene_text", "")
    hits = {mode: sum(1 for kw in keywords if kw in text) for mode, keywords in KEYWORDS.items()}
    suggested = sorted(hits.items(), key=lambda x: x[1], reverse=True)

    # 检查是否有 runtime artifacts
    has_runtime = any(f in data for f in RUNTIME_ARTIFACTS)
    if has_runtime:
        suggested = [("runtime_state", 10)] + suggested

    # 取最高分
    top_mode, top_score = suggested[0]
    if top_score == 0 and not has_runtime:
        return None
    return top_mode


def main():
    candidates = []
    for json_file in CANDIDATES_DIR.glob("candidate_*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
            data["_file"] = json_file
            candidates.append(data)

    print(f"Loaded {len(candidates)} candidates")

    # 按类别分组
    by_mode = defaultdict(list)
    for c in candidates:
        mode = classify_candidate(c)
        if mode:
            by_mode[mode].append(c)

    # 配额选择
    selected = []
    used_ids = set()

    for mode, target in TARGET_COUNTS.items():
        available = by_mode.get(mode, [])
        # 按长度排序，优先选择较长的
        available.sort(key=lambda x: len(x["scene_text"]), reverse=True)
        count = 0
        for c in available:
            if count >= target:
                break
            # 避免完全重复（通过 scene_text 去重）
            if c["scene_text"] in used_ids:
                continue
            used_ids.add(c["scene_text"])
            selected.append((mode, c))
            count += 1
        print(f"{mode}: selected {count}/{target}")

    # 如果某类不足，从其他类别补充（但保持类别标签）
    # 这里简单处理：如果总数不足 25，从场景过渡中补充
    if len(selected) < 25:
        extra = by_mode.get("scene_transition", [])
        for c in extra:
            if len(selected) >= 25:
                break
            if c["scene_text"] not in used_ids:
                used_ids.add(c["scene_text"])
                selected.append(("scene_transition", c))

    print(f"\nTotal selected: {len(selected)}")

    # 输出待确认列表
    print("\nReview the following candidates:")
    for i, (mode, c) in enumerate(selected):
        print(f"[{i+1}] {mode}: {c['_file'].name}")
        print(f"    Preview: {c['scene_text'][:100]}...")
        print()

    # 交互式确认（可选）
    confirm = input("Accept all? (y/n): ").strip().lower()
    if confirm != "y":
        print("Skipping automatic generation. You can edit the selected list manually.")
        return

    # 生成 YAML
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = []
    for mode, c in selected:
        sample_id = f"corpus.{mode}.manual.{c['id']}"
        difficulty = "medium"
        if len(c["scene_text"]) < 200:
            difficulty = "easy"
        elif "禁地" in c["scene_text"] or "心魔" in c["scene_text"]:
            difficulty = "hard"

        sample = {
            "id": sample_id,
            "version": "1.0",
            "category": mode,
            "failure_modes": [mode],
            "difficulty": difficulty,
            "language": "zh-CN",
            "scene_before": c["scene_text"],
            "scene_after": None,
            "source": "manual",
            "license": "internal",
            "tags": ["manual", "phase12.1b"],
            "expected": {},
            "artifacts": {},
        }

        # 如果候选包含 runtime artifacts，可以保留
        for field in RUNTIME_ARTIFACTS:
            if field in c and c[field]:
                sample["artifacts"][field] = c[field]

        samples.append(sample)

    # 写入 YAML
    manifest = {"version": "1.0", "created_at": "2026-07-27T00:00:00", "categories": [], "samples": []}
    categories = set()
    for sample in samples:
        category = sample["category"]
        categories.add(category)
        category_dir = OUTPUT_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = category_dir / f"{sample['id']}.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(sample, f, allow_unicode=True, sort_keys=False)
        manifest["samples"].append({"path": str(yaml_path.relative_to(OUTPUT_DIR))})

    manifest["categories"] = list(categories)
    manifest["total_samples"] = len(samples)

    with open(OUTPUT_DIR / "corpus.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, allow_unicode=True, sort_keys=False)

    print(f"\n✅ Generated {len(samples)} samples to {OUTPUT_DIR}")
    print("Distribution:")
    for mode in TARGET_COUNTS.keys():
        count = sum(1 for s in samples if s["category"] == mode)
        print(f"  {mode}: {count}")


if __name__ == "__main__":
    main()