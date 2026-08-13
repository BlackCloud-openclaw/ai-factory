#!/usr/bin/env python3
"""
交互式标注候选场景
按 y 保留，按 n 跳过，然后选择 FailureMode 和 Difficulty
"""

import json
import shutil
from pathlib import Path

CANDIDATES_DIR = Path("experiments/phase12/corpus/candidates")
OUTPUT_DIR = Path("experiments/phase12/corpus/v1.0")

# 定义选项
FAILURE_MODES = [
    "scene_transition",
    "planning_execution",
    "runtime_state",
    "character_state",
    "dialogue_quality",
]

DIFFICULTIES = ["easy", "medium", "hard"]


def main():
    annotated = []
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for json_file in sorted(CANDIDATES_DIR.glob("candidate_*.json")):
        with open(json_file, "r") as f:
            data = json.load(f)

        scene_text = data["scene_text"]
        print("\n" + "=" * 70)
        print(f"File: {json_file.name}")
        print(f"Chapter {data['chapter']}, Scene {data['scene_idx']}, Version {data['version']}")
        print("-" * 40)
        print(scene_text[:500] + ("..." if len(scene_text) > 500 else ""))
        print("-" * 40)

        keep = input("Keep this sample? (y/n/skip): ").strip().lower()
        if keep == "n" or keep == "skip":
            continue

        print("\nFailure Modes:")
        for i, fm in enumerate(FAILURE_MODES, 1):
            print(f"  {i}. {fm}")
        fm_choice = int(input("Select failure mode (number): ").strip())
        failure_mode = FAILURE_MODES[fm_choice - 1]

        print("\nDifficulties:")
        for i, d in enumerate(DIFFICULTIES, 1):
            print(f"  {i}. {d}")
        d_choice = int(input("Select difficulty (number): ").strip())
        difficulty = DIFFICULTIES[d_choice - 1]

        # 生成样本 ID
        sample_id = f"corpus.{failure_mode}.manual.{data['id']}"

        annotated.append({
            "id": sample_id,
            "version": "1.0",
            "category": failure_mode,
            "failure_modes": [failure_mode],
            "difficulty": difficulty,
            "language": "zh-CN",
            "scene_before": scene_text,
            "scene_after": None,
            "source": "manual",
            "license": "internal",
            "tags": ["manual", "phase12.1a4"],
            "expected": {},  # 稍后补充
            "artifacts": {},
        })

        print(f"\n✅ Annotated: {sample_id}")

    print(f"\nAnnotated {len(annotated)} samples.")

    # 保存标注结果
    with open("annotated_samples.json", "w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2, ensure_ascii=False)
    print("Saved to annotated_samples.json")


if __name__ == "__main__":
    main()