#!/usr/bin/env python3
"""
生成 Metrics Distribution Report
用于后续 expected 基线的制定
"""

import json
import statistics
from pathlib import Path
from collections import defaultdict

RESULTS_PATH = Path("experiments/phase12/results/latest.json")


def main():
    if not RESULTS_PATH.exists():
        print("❌ latest.json not found. Run benchmark first.")
        return

    with open(RESULTS_PATH, "r") as f:
        data = json.load(f)

    # 按指标分组
    scores_by_metric = defaultdict(list)

    for mr in data.get("metric_results", []):
        name = mr["name"]
        score = mr["score"]
        if score is not None:
            scores_by_metric[name].append(score)

    print("\n" + "=" * 60)
    print("📊 Metric Distribution Report")
    print("=" * 60)

    # 排序输出
    for metric, scores in sorted(scores_by_metric.items()):
        if not scores:
            continue

        n = len(scores)
        mean = statistics.mean(scores)
        median = statistics.median(scores)
        if len(scores) >= 2:
            stdev = statistics.stdev(scores)
        else:
            stdev = 0.0
        min_score = min(scores)
        max_score = max(scores)

        print(f"\n📈 {metric}")
        print(f"   samples: {n}")
        print(f"   mean:    {mean:.4f}")
        print(f"   median:  {median:.4f}")
        print(f"   std:     {stdev:.4f}")
        print(f"   min:     {min_score:.4f}")
        print(f"   max:     {max_score:.4f}")

    # 汇总表
    print("\n" + "=" * 60)
    print("📋 Summary Table")
    print("=" * 60)
    print(f"{'Metric':<22} {'Count':>6} {'Mean':>8} {'Min':>8} {'Max':>8} {'Std':>8}")
    print("-" * 62)

    for metric, scores in sorted(scores_by_metric.items()):
        if not scores:
            continue
        n = len(scores)
        mean = statistics.mean(scores)
        min_score = min(scores)
        max_score = max(scores)
        stdev = statistics.stdev(scores) if n >= 2 else 0.0
        print(f"{metric:<22} {n:>6} {mean:>8.3f} {min_score:>8.3f} {max_score:>8.3f} {stdev:>8.3f}")

    print("=" * 60)


if __name__ == "__main__":
    main()