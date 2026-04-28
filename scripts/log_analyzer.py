#!/usr/bin/env python
"""
Log Analyzer for AI Factory
Reads structured JSON logs and outputs performance metrics.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

LOG_FILE = Path("logs/ai_factory.log")


def load_logs():
    """Yield each valid JSON log line."""
    if not LOG_FILE.exists():
        print(f"Log file not found: {LOG_FILE}")
        return
    with open(LOG_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass  # skip malformed lines


def percentile(data, p):
    """Calculate the p-th percentile (0-100) of a list of numbers."""
    if not data:
        return None
    sorted_data = sorted(data)
    n = len(sorted_data)
    rank = (p / 100.0) * (n - 1)
    lower = int(rank)
    upper = lower + 1
    if upper >= n:
        return sorted_data[lower]
    weight = rank - lower
    return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight


def analyze():
    agent_durations = defaultdict(list)
    agent_errors = defaultdict(int)
    step_counts = defaultdict(list)
    total_requests = 0
    success_requests = 0
    all_durations = []   # 收集所有请求的耗时用于 P99

    for log in load_logs():
        if log.get("level") == "INFO" and "completed" in log.get("message", ""):
            agent = log.get("agent")
            duration = log.get("duration")
            if agent and duration is not None:
                agent_durations[agent].append(duration)
                all_durations.append(duration)
                total_requests += 1
                if log.get("status") == "success":
                    success_requests += 1
        elif log.get("level") == "ERROR":
            agent = log.get("agent")
            if agent:
                agent_errors[agent] += 1

        step = log.get("step")
        if step is not None:
            step_counts[step].append(1)

    print("\n=== Agent Average Duration (seconds) ===")
    for agent in sorted(agent_durations.keys()):
        avg = sum(agent_durations[agent]) / len(agent_durations[agent])
        print(f"{agent:20} : {avg:.3f}s (count={len(agent_durations[agent])})")

    print("\n=== Agent Error Count ===")
    for agent, count in sorted(agent_errors.items(), key=lambda x: x[1], reverse=True):
        print(f"{agent:20} : {count}")

    print("\n=== Step Distribution ===")
    for step in sorted(step_counts.keys()):
        print(f"Step {step:2} : {len(step_counts[step])} occurrences")

    # 计算全局 P99 耗时
    if all_durations:
        p99 = percentile(all_durations, 99)
        print(f"\n=== Overall ===")
        print(f"Total completed requests: {total_requests}")
        print(f"Success rate: {success_requests/total_requests*100:.1f}%" if total_requests else "N/A")
        print(f"P99 duration: {p99:.2f}s")
    else:
        print("\n=== Overall ===")
        print("No completed requests found.")


if __name__ == "__main__":
    analyze()