#!/usr/bin/env python3
"""
自动重建 Gold Corpus v1.1（增强去重 + 格式统一）

用法：
    python scripts/rebuild_corpus.py --force   # 强制重建
    python scripts/rebuild_corpus.py --dry-run # 预览
"""

import json
import yaml
import shutil
import hashlib
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple
import argparse
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 目录配置
CANDIDATES_DIR = Path("experiments/phase12/corpus/candidates")
OUTPUT_DIR = Path("experiments/phase12/corpus/v1.0")
BACKUP_DIR = Path("experiments/phase12/corpus/v1.0.bak")

# 目标配额
TARGET = {
    "scene_transition": 5,
    "character_state": 5,
    "dialogue_quality": 5,
    "planning_execution": 5,
    "runtime_state": 5,
}
TOTAL = sum(TARGET.values())  # 25

# 分类关键词
KEYWORDS = {
    "scene_transition": [
        "时间", "突然", "第二天", "转向", "切换", "转移", "进入", "离开",
        "来到", "踏进", "走出", "穿过", "越过", "抵达", "返回"
    ],
    "character_state": [
        "境界", "突破", "金丹", "元婴", "修为", "灵力", "血脉", "伤势",
        "重伤", "轻伤", "恢复", "痊愈", "经脉", "丹田", "神魂"
    ],
    "dialogue_quality": [
        "对话", "说", "问", "答", "道", "喊", "叫", "交谈", "质问",
        "冷笑", "低喝", "咆哮", "呢喃", "叹息", "怒吼"
    ],
    "planning_execution": [
        "计划", "目标", "执行", "任务", "完成", "必须", "契约", "规划",
        "安排", "步骤", "方案", "布置", "准备", "行动"
    ],
    "runtime_state": [
        "状态", "快照", "事件", "记录", "更新", "日志", "数据",
        "runtime", "metrics", "snapshot", "artifact"
    ],
}
RUNTIME_ARTIFACTS = {"runtime_metrics", "snapshot_before", "snapshot_after", "events"}


# ========== 文本归一化（用于去重） ==========
def normalize_text(text: str) -> str:
    """去除空白符、统一标点，用于去重哈希"""
    if not text:
        return ""
    # 去除所有空白（包括换行、空格、制表）
    text = re.sub(r'\s+', '', text)
    # 统一标点：将「」“”转换为普通引号（可选）
    # 这里不转换，保留原样，但去除空白已足够
    return text


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()[:16]


# ========== 内容相似度计算（基于单词集合） ==========
def jaccard_similarity(text1: str, text2: str, threshold: float = 0.6) -> float:
    """计算两段文本的 Jaccard 相似度（基于中文分词）"""
    # 简单分词：按非中文分割
    words1 = set(re.findall(r'[\u4e00-\u9fff]+', text1))
    words2 = set(re.findall(r'[\u4e00-\u9fff]+', text2))
    if not words1 or not words2:
        return 0.0
    inter = len(words1 & words2)
    union = len(words1 | words2)
    return inter / union if union else 0.0


# ========== 分类与评分 ==========
def classify_candidate(data: Dict) -> Tuple[str, float]:
    """返回 (最佳类别, 置信度分数)"""
    text = data.get("scene_text", "")
    if not text:
        return ("unknown", 0.0)

    # 检查 runtime artifacts
    has_runtime = any(f in data for f in RUNTIME_ARTIFACTS)

    # 关键词匹配
    scores = {}
    for mode, keywords in KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text)
        scores[mode] = count

    # runtime_state 加分
    if has_runtime:
        scores["runtime_state"] += 3.0

    best = max(scores.items(), key=lambda x: x[1])
    return (best[0], best[1])


def quality_score(data: Dict) -> float:
    """综合质量分"""
    text = data.get("scene_text", "")
    length_score = min(len(text) / 500, 1.0) * 2
    dialogue_quotes = text.count("「") + text.count("」") + text.count('"')
    dialogue_score = min(dialogue_quotes / 10, 1.0) * 1.5
    runtime_bonus = 1.0 if any(f in data for f in RUNTIME_ARTIFACTS) else 0.0
    class_score = data.get("_class_score", 0) * 0.5
    return length_score + dialogue_score + runtime_bonus + class_score


# ========== 加载候选 ==========
def load_candidates() -> List[Dict]:
    candidates = []
    for json_file in sorted(CANDIDATES_DIR.glob("candidate_*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["_file"] = json_file.name
            candidates.append(data)
    return candidates


# ========== 选择样本 ==========
def select_samples(candidates: List[Dict], target: Dict[str, int]) -> Dict[str, List[Dict]]:
    # 1. 基于归一化哈希去重
    seen_hashes = set()
    unique = []
    for c in candidates:
        h = text_hash(c.get("scene_text", ""))
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(c)

    print(f"去重后候选数: {len(unique)} (原 {len(candidates)})")

    # 2. 分类并评分
    classified = defaultdict(list)
    for c in unique:
        mode, score = classify_candidate(c)
        if mode == "unknown":
            continue
        c["_class_score"] = score
        classified[mode].append(c)

    # 3. 按类别选择，并防止内容相似
    selected = defaultdict(list)
    selected_texts = []  # 用于相似度过滤

    for mode, count_needed in target.items():
        pool = classified.get(mode, [])
        # 按质量分排序
        pool.sort(key=quality_score, reverse=True)

        added = 0
        for c in pool:
            if added >= count_needed:
                break
            # 检查与已选样本的相似度（仅当已选样本数>0）
            text = c.get("scene_text", "")
            duplicate = False
            for existing in selected_texts:
                if jaccard_similarity(text, existing) > 0.6:  # 阈值
                    duplicate = True
                    break
            if duplicate:
                continue
            # 加入选择
            selected[mode].append(c)
            selected_texts.append(text)
            added += 1

        # 如果不足，从所有未选候选中补充（强制类别）
        if added < count_needed:
            deficit = count_needed - added
            all_unselected = [c for c in unique if not any(c in selected[m] for m in selected)]
            all_unselected.sort(key=quality_score, reverse=True)
            for c in all_unselected:
                if deficit <= 0:
                    break
                # 再次检查相似度
                text = c.get("scene_text", "")
                duplicate = False
                for existing in selected_texts:
                    if jaccard_similarity(text, existing) > 0.6:
                        duplicate = True
                        break
                if duplicate:
                    continue
                c["_forced_mode"] = mode
                selected[mode].append(c)
                selected_texts.append(text)
                deficit -= 1

        print(f"{mode}: 候选 {len(pool)} 个，选择 {len(selected[mode])} 个")

    return selected


# ========== 生成 YAML ==========
def generate_yaml(sample_data: Dict, category: str, index: int) -> Dict:
    """生成单个 YAML，并确保 scene_before 为纯文本"""
    raw = sample_data.get("scene_text", "")

    # 尝试解析 JSON 并提取 scene_text
    scene_text = raw
    artifacts = {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            if "scene_text" in parsed:
                scene_text = parsed["scene_text"]
            # 提取 events / foreshadowing
            if "events" in parsed and parsed["events"]:
                artifacts["events"] = parsed["events"]
            if "foreshadowing" in parsed and parsed["foreshadowing"]:
                artifacts["foreshadowing"] = parsed["foreshadowing"]
        elif isinstance(parsed, list):
            # 如果数组，合并
            scene_text = "\n".join(str(item) for item in parsed)
    except json.JSONDecodeError:
        # 已经是纯文本
        pass

    # 额外复制 runtime artifacts 字段（如果存在）
    for field in RUNTIME_ARTIFACTS:
        if field in sample_data and sample_data[field]:
            artifacts[field] = sample_data[field]

    # 难度判断
    difficulty = "medium"
    if len(scene_text) < 200:
        difficulty = "easy"
    elif any(kw in scene_text for kw in ["禁地", "心魔", "金丹", "元婴", "阵眼", "血瞳"]):
        difficulty = "hard"

    sample_id = f"corpus.{category}.auto.{index:03d}"

    return {
        "id": sample_id,
        "version": "1.0",
        "category": category,
        "failure_modes": [category],
        "difficulty": difficulty,
        "language": "zh-CN",
        "scene_before": scene_text,
        "scene_after": None,
        "source": "auto",
        "license": "internal",
        "tags": ["auto", "phase12.1b", f"category_{category}"],
        "expected": {},
        "artifacts": artifacts,
    }


def rebuild_corpus(force: bool = False, dry_run: bool = False):
    if force and not dry_run:
        if OUTPUT_DIR.exists():
            if BACKUP_DIR.exists():
                shutil.rmtree(BACKUP_DIR)
            shutil.move(OUTPUT_DIR, BACKUP_DIR)
            print(f"✅ 已备份旧 Corpus 到 {BACKUP_DIR}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    elif not dry_run:
        if OUTPUT_DIR.exists():
            print(f"⚠️ {OUTPUT_DIR} 已存在，使用 --force 覆盖或 --dry-run 预览")
            return

    candidates = load_candidates()
    print(f"📂 加载候选场景: {len(candidates)}")

    selected = select_samples(candidates, TARGET)

    # 生成 YAML
    all_samples = []
    for category, samples in selected.items():
        for idx, sample_data in enumerate(samples, 1):
            yaml_data = generate_yaml(sample_data, category, idx)
            all_samples.append(yaml_data)

    if dry_run:
        print("\n📋 预览选择结果:")
        for category, samples in selected.items():
            print(f"  {category}: {len(samples)} 个")
            for s in samples:
                print(f"    - {s['_file']} (score={s.get('_class_score',0):.1f})")
        return

    # 写入文件
    samples_written = 0
    categories_set = set()
    manifest_path = OUTPUT_DIR / "corpus.yaml"

    for yaml_data in all_samples:
        category = yaml_data["category"]
        categories_set.add(category)
        category_dir = OUTPUT_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = category_dir / f"{yaml_data['id']}.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_data, f, allow_unicode=True, sort_keys=False)
        samples_written += 1

    # Manifest
    manifest = {
        "version": "1.0",
        "created_at": "2026-07-27T00:00:00",
        "categories": sorted(categories_set),
        "samples": [
            {"path": f"{s['category']}/{s['id']}.yaml"}
            for s in all_samples
        ],
        "total_samples": samples_written,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, allow_unicode=True, sort_keys=False)

    print(f"\n✅ Corpus 重建完成！")
    print(f"   总样本: {samples_written}")
    print(f"   类别: {', '.join(categories_set)}")
    print(f"   输出目录: {OUTPUT_DIR}")

    # 统计分布
    dist = defaultdict(int)
    for s in all_samples:
        dist[s["category"]] += 1
    print("\n📊 最终分布:")
    for cat in sorted(dist):
        print(f"   {cat}: {dist[cat]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动重建 Gold Corpus")
    parser.add_argument("--force", action="store_true", help="强制覆盖现有 v1.0 目录")
    parser.add_argument("--dry-run", action="store_true", help="只预览，不写入文件")
    args = parser.parse_args()
    rebuild_corpus(force=args.force, dry_run=args.dry_run)