#!/usr/bin/env python
"""
Log Analyzer for AI Factory
Reads structured JSON logs and outputs performance metrics.
"""

import json
import sys
import re
from collections import defaultdict, Counter
from pathlib import Path

LOG_FILE = Path("logs/ai_factory.log")


def load_logs():
    """Yield each valid JSON log line."""
    if not LOG_FILE.exists():
        print(f"Log file not found: {LOG_FILE}")
        return
    with open(LOG_FILE, "r", encoding="utf-8") as f:
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
    error_messages = Counter()
    step_counts = defaultdict(int)
    total_requests = 0
    success_requests = 0
    all_durations = []

    duration_re = re.compile(r'duration=([0-9.]+)')
    step_re = re.compile(r'step=([0-9]+)')

    for log in load_logs():
        level = log.get("level")
        name = log.get("name", "")
        message = log.get("message", "")

        # Extract agent from name (e.g., "agents.memory" -> "memory")
        if name.startswith("agents."):
            agent = name[7:]  # remove "agents."
        else:
            agent = name

        # Extract step from message if present
        step_match = step_re.search(message)
        if step_match:
            step = int(step_match.group(1))
            step_counts[step] += 1

        # Look for completion messages
        if level == "INFO" and "completed" in message:
            # Extract duration
            dur_match = duration_re.search(message)
            if dur_match:
                try:
                    duration = float(dur_match.group(1))
                    agent_durations[agent].append(duration)
                    all_durations.append(duration)
                    total_requests += 1
                    if "success" in message:
                        success_requests += 1
                except ValueError:
                    pass

        # Error logs
        elif level == "ERROR":
            agent_errors[agent] += 1
            # Truncate message for grouping
            error_type = message[:50] if message else "empty error"
            error_messages[error_type] += 1

    print("\n=== Agent Average Duration (seconds) ===")
    if agent_durations:
        for agent in sorted(agent_durations.keys()):
            durations = agent_durations[agent]
            avg = sum(durations) / len(durations)
            print(f"{agent:20} : {avg:.3f}s (count={len(durations)})")
    else:
        print("No agent duration data found.")

    print("\n=== Agent Error Count ===")
    if agent_errors:
        for agent, count in sorted(agent_errors.items(), key=lambda x: x[1], reverse=True):
            print(f"{agent:20} : {count}")
    else:
        print("No errors found.")

    if error_messages:
        print("\n=== Top 10 Error Messages ===")
        for msg, count in error_messages.most_common(10):
            print(f"{count:>5} : {msg}")

    if step_counts:
        print("\n=== Step Distribution (occurrences) ===")
        for step in sorted(step_counts.keys()):
            print(f"Step {step:2} : {step_counts[step]}")

    if all_durations:
        p99 = percentile(all_durations, 99)
        print(f"\n=== Overall ===")
        print(f"Total completed requests: {total_requests}")
        print(f"Success rate: {success_requests/total_requests*100:.1f}%" if total_requests else "N/A")
        print(f"P99 duration: {p99:.2f}s")
    else:
        print("\n=== Overall ===")
        print("No completed requests found with duration.")


if __name__ == "__main__":
    analyze()