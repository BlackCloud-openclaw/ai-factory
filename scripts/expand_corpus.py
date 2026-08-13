#!/usr/bin/env python3
"""
从候选场景中自动选择缺失类别，扩充 Corpus
"""

import json
import yaml
from pathlib import Path
from collections import defaultdict

CANDIDATES_DIR = Path("experiments/phase12/corpus/candidates")
OUTPUT_DIR = Path("experiments/phase12/corpus/v1.0")


def classify_failure_mode(text):
    """根据文本内容分配 Failure Mode"""
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["计划", "目标", "必须", "执行", "任务", "完成", "契约"]):
        return "planning_execution"
    elif any(kw in text_lower for kw in ["状态", "快照", "事件", "记录", "更新", "日志"]):
        return "runtime_state"
    elif any(kw in text_lower for kw in ["修为", "境界", "突破", "金丹", "灵力", "元婴", "血脉"]):
        return "character_state"
    elif any(kw in text_lower for kw in ["对话", "说", "问", "答", "道", "问", "喊", "叫"]):
        return "dialogue_quality"
    elif any(kw in text_lower for kw in ["时间", "突然", "第二天", "转向", "切换", "转移", "场景"]):
        return "scene_transition"
    return "runtime_state"


def assign_difficulty(text, length):
    if len(text) < 200:
        return "easy"
    if any(kw in text for kw in ["禁地", "心魔", "密信", "金丹碎片", "阵眼", "血瞳"]):
        return "hard"
    return "medium"


def main():
    # 加载所有候选
    candidates = []
    for json_file in sorted(CANDIDATES_DIR.glob("candidate_*.json")):
        with open(json_file, "r") as f:
            data = json.load(f)
            data["_file"] = json_file.name
            candidates.append(data)

    # 去重（同场景多版本保留一个）
    grouped = defaultdict(list)
    for c in candidates:
        key = (c["volume"], c["chapter"], c["scene_idx"])
        grouped[key].append(c)

    version_priority = {"C": 2, "B": 1, "A": 0}
    unique = []
    for key, items in grouped.items():
        items.sort(key=lambda x: version_priority.get(x["version"], -1), reverse=True)
        unique.append(items[0])

    print(f"Loaded {len(candidates)} candidates, {len(unique)} unique scenes")

    # 按类别分组
    by_mode = defaultdict(list)
    for c in unique:
        fm = classify_failure_mode(c["scene_text"])
        by_mode[fm].append(c)

    # 目标数量
    target = {
        "scene_transition": 4,
        "character_state": 4,
        "planning_execution": 4,
        "runtime_state": 4,
        "dialogue_quality": 4,
    }

    # 加载已存在的样本
    existing = {}
    for mode_dir in OUTPUT_DIR.iterdir():
        if mode_dir.is_dir():
            for yaml_file in mode_dir.glob("*.yaml"):
                with open(yaml_file, "r") as f:
                    sample = yaml.safe_load(f)
                    existing[sample["id"]] = sample

    # 统计已存在的类别
    existing_count = defaultdict(int)
    for sample in existing.values():
        existing_count[sample["category"]] += 1

    print("\nCurrent distribution:")
    for fm in target.keys():
        print(f"  {fm}: {existing_count[fm]}")

    # 补充缺失的类别
    samples_to_add = []
    for fm, count_needed in target.items():
        current = existing_count[fm]
        need = count_needed - current
        if need <= 0:
            continue

        available = by_mode.get(fm, [])
        # 按难度和长度排序，优先选中等及以上
        available.sort(key=lambda x: len(x["scene_text"]), reverse=True)

        added = 0
        for c in available:
            if added >= need:
                break
            # 检查是否已被选择（通过 scene_text 去重）
            text = c["scene_text"]
            is_duplicate = any(
                sample.get("scene_before", "") == text
                for sample in existing.values()
            )
            if is_duplicate:
                continue

            # 生成样本
            sample_id = f"corpus.{fm}.manual.{c['id']}"
            difficulty = assign_difficulty(text, len(text))
            sample = {
                "id": sample_id,
                "version": "1.0",
                "category": fm,
                "failure_modes": [fm],
                "difficulty": difficulty,
                "language": "zh-CN",
                "scene_before": text,
                "scene_after": None,
                "source": "manual",
                "license": "internal",
                "tags": ["manual", "phase12.1a5"],
                "expected": {},
                "artifacts": {},
            }
            samples_to_add.append(sample)
            existing_count[fm] += 1
            added += 1
            print(f"  Added {fm}: {sample_id}")

    if not samples_to_add:
        print("\nNo new samples needed, already balanced.")
        return

    # 保存新样本
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for sample in samples_to_add:
        category = sample["category"]
        category_dir = OUTPUT_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = category_dir / f"{sample['id']}.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(sample, f, allow_unicode=True, sort_keys=False)

    # 更新 manifest
    manifest_path = OUTPUT_DIR / "corpus.yaml"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = yaml.safe_load(f)
    else:
        manifest = {"version": "1.0", "created_at": "2026-07-24T00:00:00", "categories": [], "samples": []}

    # 合并已有样本路径
    existing_paths = {entry["path"] for entry in manifest.get("samples", [])}
    for sample in samples_to_add:
        path = f"{sample['category']}/{sample['id']}.yaml"
        if path not in existing_paths:
            manifest["samples"].append({"path": path})
            existing_paths.add(path)

    categories = set()
    for sample in manifest["samples"]:
        categories.add(sample["path"].split("/")[0])
    manifest["categories"] = list(categories)
    manifest["total_samples"] = len(manifest["samples"])

    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, allow_unicode=True, sort_keys=False)

    print(f"\n✅ Added {len(samples_to_add)} samples. Total: {len(manifest['samples'])}")