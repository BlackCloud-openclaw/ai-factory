#!/usr/bin/env python3
"""
统一 Corpus 格式：将 JSON 格式的 scene_before 转换为纯文本
提取 events/foreshadowing 到 artifacts
"""

import json
import yaml
from pathlib import Path

CORPUS_DIR = Path("experiments/phase12/corpus/v1.0")


def process_yaml(yaml_path: Path):
    """处理单个 YAML 文件"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    scene_before = data.get("scene_before", "")
    if not scene_before:
        return

    stripped = scene_before.strip()
    if not stripped.startswith(("{", "[")):
        return  # 已经是纯文本

    try:
        parsed = json.loads(scene_before)
        if isinstance(parsed, dict) and "scene_text" in parsed:
            data["scene_before"] = parsed["scene_text"]
            # 迁移到 artifacts
            artifacts = data.get("artifacts") or {}
            if "events" in parsed and parsed["events"]:
                artifacts["events"] = parsed["events"]
            if "foreshadowing" in parsed and parsed["foreshadowing"]:
                artifacts["foreshadowing"] = parsed["foreshadowing"]
            data["artifacts"] = artifacts
            # 写回
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
            print(f"✅ 转换: {yaml_path.name}")
    except json.JSONDecodeError:
        pass  # 保持原样


def main():
    count = 0
    for yaml_path in CORPUS_DIR.glob("**/*.yaml"):
        if yaml_path.name == "corpus.yaml":
            continue
        process_yaml(yaml_path)
        count += 1
    print(f"\n✅ 处理完成，共检查 {count} 个样本")
    print(f"📁 目录: {CORPUS_DIR}")


if __name__ == "__main__":
    main()