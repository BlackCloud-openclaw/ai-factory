#!/usr/bin/env python3
"""
修复无法解析的 JSON 格式残留
使用正则提取 scene_text
"""

import re
import yaml
from pathlib import Path

CORPUS_DIR = Path("experiments/phase12/corpus/v1.0")


def extract_scene_text_robust(raw: str) -> str:
    """
    使用正则从可能包含 JSON 的字符串中提取 scene_text
    即使 JSON 格式不完整也能工作
    """
    # 尝试匹配 "scene_text": "..."
    # 支持多行和转义字符
    match = re.search(r'"scene_text"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
    if match:
        text = match.group(1)
        # 处理转义字符
        text = text.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
        return text

    # 如果上面的不匹配，尝试更宽松的匹配：scene_text: 后跟任意内容直到下一个键
    match = re.search(r'"scene_text"\s*:\s*"([^"]+)"', raw, re.DOTALL)
    if match:
        return match.group(1)

    # 如果都失败，返回原样
    return raw


def main():
    count = 0
    for yaml_path in CORPUS_DIR.glob("**/*.yaml"):
        if yaml_path.name == "corpus.yaml":
            continue

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        scene_before = data.get("scene_before", "")
        if not scene_before:
            continue

        stripped = scene_before.strip()
        # 如果以 { 或 [ 开头，但解析可能失败，我们用正则提取
        if stripped.startswith(("{", "[")):
            new_text = extract_scene_text_robust(scene_before)
            # 如果提取成功且不是空
            if new_text and new_text != scene_before:
                data["scene_before"] = new_text
                with open(yaml_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
                print(f"✅ 修复: {yaml_path.name}")
                count += 1

    print(f"\n✅ 共修复 {count} 个文件")


if __name__ == "__main__":
    main()