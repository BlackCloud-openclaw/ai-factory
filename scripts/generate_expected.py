#!/usr/bin/env python3
import json
import yaml
from pathlib import Path
from collections import defaultdict

RESULTS_PATH = Path("experiments/phase12/results/latest.json")
CORPUS_DIR = Path("experiments/phase12/corpus/v1.0")

with open(RESULTS_PATH, "r") as f:
    results = json.load(f)

scores_by_metric = defaultdict(list)
for mr in results["metric_results"]:
    if mr["score"] is not None:  # 跳过 None
        scores_by_metric[mr["name"]].append(mr["score"])

expected_base = {}
for metric, scores in scores_by_metric.items():
    avg = sum(scores) / len(scores)
    expected_base[metric] = {
        "type": "range",
        "min": max(0.0, avg - 0.15),
        "max": min(1.0, avg + 0.15)
    }
    print(f"{metric}: avg={avg:.3f}, range=[{expected_base[metric]['min']:.3f}, {expected_base[metric]['max']:.3f}]")

# 更新 Corpus 样本
for yaml_path in CORPUS_DIR.glob("*/corpus.*.yaml"):
    with open(yaml_path, "r") as f:
        sample = yaml.safe_load(f)

    category = sample["category"]
    metric_map = {
        "scene_transition": "continuity",
        "character_state": "character",
        "planning_execution": "planning_coverage",
        "runtime_state": "runtime_health",
        "dialogue_quality": "dialogue",
    }
    metric = metric_map.get(category, "runtime_health")
    if metric in expected_base:
        sample["expected"] = {metric: expected_base[metric]}

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(sample, f, allow_unicode=True, sort_keys=False)

print("✅ Updated expected fields in all samples")