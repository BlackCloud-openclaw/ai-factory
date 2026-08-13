#!/usr/bin/env python3
"""
检查 Corpus v1.1 是否满足冻结条件

输出：
- 各 Failure Mode 分布
- 样本独立性检查（hash 去重）
- 格式是否统一（JSON vs 纯文本）
- runtime_state 样本是否包含真正的状态语义
"""

import json
import yaml
import re
from pathlib import Path
from collections import defaultdict

CORPUS_DIR = Path("experiments/phase12/corpus/v1.0")

# runtime_state 应该包含的语义关键词
RUNTIME_SEMANTIC_KEYWORDS = [
    "状态", "变化", "异常", "漂移", "不一致", "错误", "失败",
    "回滚", "重试", "验证", "不一致", "矛盾", "冲突"
]


def normalize(text):
    return re.sub(r'\s+', '', text)


def is_pure_text(text):
    """检查是否为纯文本（而非 JSON）"""
    if not text:
        return True
    stripped = text.strip()
    return not stripped.startswith(("{", "["))


def is_runtime_state_sample(text):
    """检查样本是否包含 runtime_state 语义"""
    return any(kw in text for kw in RUNTIME_SEMANTIC_KEYWORDS)


def main():
    manifest_path = CORPUS_DIR / "corpus.yaml"
    if not manifest_path.exists():
        print(f"❌ manifest 不存在: {manifest_path}")
        return

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    print("=" * 60)
    print("Phase 12.1B Freeze Checklist")
    print("=" * 60)

    # 1. 样本数
    samples = manifest.get("samples", [])
    print(f"\n📊 总样本数: {len(samples)}/25")

    # 2. 类别分布
    dist = defaultdict(int)
    json_count = 0
    dup_count = 0
    seen_hashes = set()
    runtime_issue = []

    for entry in samples:
        yaml_path = CORPUS_DIR / entry["path"]
        if not yaml_path.exists():
            print(f"❌ 文件缺失: {entry['path']}")
            continue

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        category = data.get("category", "unknown")
        dist[category] += 1

        scene_before = data.get("scene_before", "")

        # 格式检查
        if not is_pure_text(scene_before):
            json_count += 1

        # 去重检查
        h = normalize(scene_before)
        if h in seen_hashes:
            dup_count += 1
            print(f"⚠️ 重复样本: {entry['path']}")
        seen_hashes.add(h)

        # runtime_state 语义检查
        if category == "runtime_state" and not is_runtime_state_sample(scene_before):
            runtime_issue.append(entry["path"])

    print("\n📈 类别分布:")
    for cat in ["scene_transition", "character_state", "dialogue_quality", "planning_execution", "runtime_state"]:
        count = dist.get(cat, 0)
        status = "✅" if count >= 5 else f"⚠️ 需补充 {5-count} 个"
        print(f"   {cat}: {count} {status}")

    # 3. 格式统一
    print(f"\n📝 格式状态:")
    print(f"   JSON 格式残留: {json_count} 个 {'⚠️ 建议统一' if json_count > 0 else '✅'}")

    # 4. 去重
    print(f"\n🔍 独立性:")
    print(f"   重复样本: {dup_count} 个 {'⚠️ 需处理' if dup_count > 0 else '✅'}")

    # 5. runtime_state 语义
    print(f"\n🎯 runtime_state 语义检查:")
    if runtime_issue:
        print(f"   ⚠️ 以下样本可能不含 runtime 语义:")
        for p in runtime_issue:
            print(f"      - {p}")
    else:
        print("   ✅ 所有 runtime_state 样本包含状态语义")

    # 结论
    print("\n" + "=" * 60)
    all_passed = (
        len(samples) >= 25 and
        all(dist.get(cat, 0) >= 5 for cat in ["scene_transition", "character_state", "dialogue_quality", "planning_execution", "runtime_state"]) and
        json_count == 0 and
        dup_count == 0 and
        not runtime_issue
    )

    if all_passed:
        print("✅ Corpus v1.1 满足冻结条件，可以进入 Phase 12.2")
    else:
        print("⚠️ 需要修复上述问题后冻结")
        print("\n推荐操作:")
        if dist.get("runtime_state", 0) < 5:
            print("  1. 补充 runtime_state 样本（需包含状态变化语义）")
        if dist.get("planning_execution", 0) < 5:
            print("  2. 补充 planning_execution 样本")
        if json_count > 0:
            print("  3. 运行 scripts/format_corpus.py 统一格式")
        if dup_count > 0:
            print("  4. 移除重复样本")
        if runtime_issue:
            print("  5. 替换不含 runtime 语义的 runtime_state 样本")

    print("=" * 60)


if __name__ == "__main__":
    main()