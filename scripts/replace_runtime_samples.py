#!/usr/bin/env python3
"""
替换 runtime_state 样本为真正包含运行时语义的候选（含去重）
"""

import json
import yaml
import shutil
import re
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

CANDIDATES_DIR = Path("experiments/phase12/corpus/candidates")
OUTPUT_DIR = Path("experiments/phase12/corpus/v1.0")
BACKUP_DIR = Path("experiments/phase12/corpus/v1.0.runtime.bak")

RUNTIME_KEYWORDS = [
    "状态", "异常", "错误", "失败", "回滚", "重试", "验证",
    "不一致", "冲突", "矛盾", "漂移", "快照", "事件", "记录",
    "runtime", "metrics", "artifact", "snapshot"
]


def normalize_text(text: str) -> str:
    """归一化用于去重"""
    return re.sub(r'\s+', '', text)


def has_runtime_semantic(text: str) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in RUNTIME_KEYWORDS)


def extract_text(data: dict) -> str:
    raw = data.get("scene_text", "")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "scene_text" in parsed:
            return parsed["scene_text"]
    except:
        pass
    return raw


def quality_score(data: dict) -> float:
    text = extract_text(data)
    length_score = min(len(text) / 500, 1.0) * 2
    dialogue_quotes = text.count("「") + text.count("」") + text.count('"')
    dialogue_score = min(dialogue_quotes / 10, 1.0) * 1.5
    keyword_count = sum(1 for kw in RUNTIME_KEYWORDS if kw in text)
    runtime_bonus = min(keyword_count / 5, 1.0) * 2
    return length_score + dialogue_score + runtime_bonus


def generate_yaml(sample_data: dict, index: int) -> dict:
    scene_before = extract_text(sample_data)
    artifacts = {}
    for field in ["runtime_metrics", "snapshot_before", "snapshot_after", "events"]:
        if field in sample_data and sample_data[field]:
            artifacts[field] = sample_data[field]
    try:
        parsed = json.loads(sample_data.get("scene_text", ""))
        if isinstance(parsed, dict):
            if "events" in parsed:
                artifacts["events"] = parsed["events"]
            if "foreshadowing" in parsed:
                artifacts["foreshadowing"] = parsed["foreshadowing"]
    except:
        pass
    difficulty = "hard" if any(kw in scene_before for kw in ["禁地", "心魔", "金丹", "元婴", "阵眼", "血瞳"]) else "medium"
    if len(scene_before) < 200:
        difficulty = "easy"
    sample_id = f"corpus.runtime_state.auto.{index:03d}"
    return {
        "id": sample_id,
        "version": "1.0",
        "category": "runtime_state",
        "failure_modes": ["runtime_state"],
        "difficulty": difficulty,
        "language": "zh-CN",
        "scene_before": scene_before,
        "scene_after": None,
        "source": "auto",
        "license": "internal",
        "tags": ["auto", "phase12.1b", "category_runtime_state"],
        "expected": {},
        "artifacts": artifacts,
    }


def main():
    runtime_dir = OUTPUT_DIR / "runtime_state"
    if runtime_dir.exists():
        if BACKUP_DIR.exists():
            shutil.rmtree(BACKUP_DIR)
        shutil.copytree(runtime_dir, BACKUP_DIR)
        print(f"✅ 已备份原 runtime_state 到 {BACKUP_DIR}")
        for f in runtime_dir.glob("*.yaml"):
            f.unlink()
    else:
        runtime_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for json_file in sorted(CANDIDATES_DIR.glob("candidate_*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["_file"] = json_file.name
            candidates.append(data)

    print(f"📂 加载候选场景: {len(candidates)}")

    runtime_candidates = []
    for c in candidates:
        text = extract_text(c)
        if has_runtime_semantic(text):
            c["_text"] = text
            runtime_candidates.append(c)

    print(f"🔍 包含 runtime 语义的候选: {len(runtime_candidates)}")

    if len(runtime_candidates) < 5:
        relaxed = [c for c in candidates if "状态" in extract_text(c) or "异常" in extract_text(c)]
        print(f"   放宽后候选: {len(relaxed)}")
        runtime_candidates = relaxed

    runtime_candidates.sort(key=quality_score, reverse=True)

    # 去重选择
    selected = []
    seen_norms = set()
    for c in runtime_candidates:
        norm = normalize_text(extract_text(c))
        if norm not in seen_norms:
            seen_norms.add(norm)
            selected.append(c)
            if len(selected) >= 5:
                break

    # 如果不足5，从剩余候选中继续补充
    if len(selected) < 5:
        for c in runtime_candidates:
            if c not in selected:
                norm = normalize_text(extract_text(c))
                if norm not in seen_norms:
                    seen_norms.add(norm)
                    selected.append(c)
                    if len(selected) >= 5:
                        break

    print(f"✅ 选中 {len(selected)} 个独立样本")

    for idx, sample_data in enumerate(selected, 1):
        yaml_data = generate_yaml(sample_data, idx)
        yaml_path = runtime_dir / f"{yaml_data['id']}.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_data, f, allow_unicode=True, sort_keys=False)
        print(f"   ✅ 生成: {yaml_path.name}")

    # 更新 manifest
    manifest_path = OUTPUT_DIR / "corpus.yaml"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f)

        new_samples = []
        for entry in manifest.get("samples", []):
            if not entry["path"].startswith("runtime_state/"):
                new_samples.append(entry)

        for idx in range(1, 6):
            new_samples.append({"path": f"runtime_state/corpus.runtime_state.auto.{idx:03d}.yaml"})

        manifest["samples"] = new_samples
        manifest["total_samples"] = len(new_samples)

        with open(manifest_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, allow_unicode=True, sort_keys=False)

        print(f"\n✅ Manifest 已更新 ({len(new_samples)} 个样本)")

    print("\n🎯 请重新运行 check_corpus_ready.py 验证")


if __name__ == "__main__":
    main()