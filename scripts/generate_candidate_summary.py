#!/usr/bin/env python3
"""
生成候选场景摘要，便于人工筛选缺失类别
"""

import json
from pathlib import Path

CANDIDATES_DIR = Path("experiments/phase12/corpus/candidates")
OUTPUT_FILE = Path("experiments/phase12/corpus/candidate_summary.txt")

# 目标类别关键词
KEYWORDS = {
    "planning_execution": ["计划", "目标", "执行", "任务", "完成", "必须", "契约", "规划"],
    "runtime_state": ["状态", "快照", "事件", "记录", "更新", "日志", "数据"],
    "dialogue_quality": ["对话", "说", "问", "答", "道", "喊", "叫", "交谈"],
    "character_state": ["境界", "突破", "金丹", "元婴", "修为", "灵力", "血脉", "伤势"],
    "scene_transition": ["时间", "突然", "第二天", "转向", "切换", "转移", "进入"],
}

# 优先关键字（用于 runtime_state 识别）
RUNTIME_ARTIFACTS = ["runtime_metrics", "snapshot_before", "snapshot_after", "events"]


def analyze_candidate(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    text = data.get("scene_text", "")
    if not text:
        return None

    # 长度
    length = len(text)

    # 关键词命中
    hits = {}
    for mode, keywords in KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text)
        hits[mode] = count

    # 检测 artifacts
    artifacts = []
    for field in RUNTIME_ARTIFACTS:
        if field in data and data[field]:
            artifacts.append(field)

    # 可能的类别（按命中数排序）
    sorted_modes = sorted(hits.items(), key=lambda x: x[1], reverse=True)
    suggested_modes = [mode for mode, count in sorted_modes if count > 0]

    # 如果命中了 runtime artifacts，优先推荐 runtime_state
    if "runtime_metrics" in artifacts or "snapshot_before" in artifacts:
        suggested_modes = ["runtime_state"] + [m for m in suggested_modes if m != "runtime_state"]

    return {
        "filename": filepath.name,
        "length": length,
        "hits": hits,
        "artifacts": artifacts,
        "suggested_modes": suggested_modes[:3],
        "preview": text[:200],
        "data": data,
    }


def main():
    summaries = []
    for json_file in sorted(CANDIDATES_DIR.glob("candidate_*.json")):
        summary = analyze_candidate(json_file)
        if summary:
            summaries.append(summary)

    # 按长度排序
    summaries.sort(key=lambda x: x["length"], reverse=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for s in summaries:
            f.write(f"=== {s['filename']} ===\n")
            f.write(f"Length: {s['length']}\n")
            f.write(f"Artifacts: {', '.join(s['artifacts']) or 'none'}\n")
            f.write(f"Suggested: {', '.join(s['suggested_modes'])}\n")
            f.write(f"Preview: {s['preview']}...\n")
            f.write("\n")

    print(f"✅ Summary written to {OUTPUT_FILE}")
    print(f"Total candidates: {len(summaries)}")


if __name__ == "__main__":
    main()