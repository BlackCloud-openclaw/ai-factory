#!/usr/bin/env python
"""
评分三版本数据（从导出 JSON 读取，或直接从数据库读取）
用法:
    python scripts/score_versions.py --input ./versions/simple_long_novel_001_v001_c001
    python scripts/score_versions.py --novel simple_long_novel_001 --chapter 1
输出: 终端打印 Markdown 表格，并可选保存 CSV/JSON
"""

import asyncio
import asyncpg
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import config
from src.writing.narrative_kpi import NarrativeKPIEngine


def score_text(text: str) -> Dict[str, float]:
    """用 NarrativeKPIEngine 对单个文本评分（无状态差异）"""
    engine = NarrativeKPIEngine()
    # 传入空状态，只基于文本特征
    result = engine.compute(text, {}, {})
    return {
        "narrative_value": result.narrative_value,
        "engagement": result.engagement,
        "progression": result.progression,
        "dialogue": result.dialogue,
        "interaction": result.interaction,
        "conflict": result.conflict,
        "pressure": result.pressure,
        "tension": result.tension,
        "relationship": result.relationship,
        "goal": result.goal,
        "character": result.character,
        "total_chars": result.total_chars,
    }


async def score_from_db(novel_id: str, volume: int, chapter: int):
    """直接从数据库读取并评分"""
    conn = await asyncpg.connect(config.postgres_dsn)
    rows = await conn.fetch("""
        SELECT scene_idx, version_type, scene_text
        FROM narrative_versions
        WHERE novel_id = $1 AND volume_num = $2 AND chapter_num = $3
        ORDER BY scene_idx, version_type
    """, novel_id, volume, chapter)

    if not rows:
        print(f"未找到数据")
        await conn.close()
        return

    # 组织数据
    scenes_data = {}
    for row in rows:
        scene_idx = row["scene_idx"]
        if scene_idx not in scenes_data:
            scenes_data[scene_idx] = {}
        scenes_data[scene_idx][row["version_type"]] = row["scene_text"]

    await conn.close()
    return score_versions(scenes_data)


def score_from_dir(input_dir: str):
    """从导出的目录读取 JSON 并评分"""
    path = Path(input_dir)
    if not path.exists() or not path.is_dir():
        print(f"目录不存在: {input_dir}")
        return

    scenes_data = {}
    for json_file in sorted(path.glob("scene_*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        scene_idx = data["scene_idx"]
        scenes_data[scene_idx] = data["versions"]

    return score_versions(scenes_data)


def score_versions(scenes_data: Dict[int, Dict[str, str]]) -> None:
    """核心评分逻辑"""
    all_scores = {}
    for scene_idx, versions in sorted(scenes_data.items()):
        scene_scores = {}
        for version_type, text in versions.items():
            # 解析 JSON（如果 text 是 JSON 字符串）
            try:
                parsed = json.loads(text)
                # 如果是完整 JSON，提取 scene_text
                if isinstance(parsed, dict) and "scene_text" in parsed:
                    scene_text = parsed["scene_text"]
                else:
                    scene_text = text
            except:
                scene_text = text

            scores = score_text(scene_text)
            scene_scores[version_type] = scores
        all_scores[scene_idx] = scene_scores

    # 打印对比报告
    print("\n📊 三版本评分对比报告\n")
    print("| 场景 | 版本 | NV | Eng | Prog | 对话 | 互动 | 冲突 | 压力 | 张力 | 关系 | 目标 | 角色 | 字数 |")
    print("|------|------|----|-----|------|------|------|------|------|------|------|------|------|------|")

    for scene_idx, scene_scores in all_scores.items():
        for version_type in ["A", "B", "C"]:
            if version_type in scene_scores:
                s = scene_scores[version_type]
                print(f"| {scene_idx:02d} | {version_type} | {s['narrative_value']:.2f} | {s['engagement']:.2f} | {s['progression']:.2f} | "
                      f"{s['dialogue']:.1f} | {s['interaction']:.1f} | {s['conflict']:.1f} | {s['pressure']:.1f} | "
                      f"{s['tension']:.1f} | {s['relationship']:.1f} | {s['goal']:.1f} | {s['character']:.1f} | {s['total_chars']} |")
        # 空行分隔场景
        print("|------|------|----|-----|------|------|------|------|------|------|------|------|------|------|")

    # 计算各版本平均分
    print("\n📈 各版本平均分\n")
    avgs = {"A": {}, "B": {}, "C": {}}
    count = {"A": 0, "B": 0, "C": 0}
    for scene_scores in all_scores.values():
        for vt in avgs.keys():
            if vt in scene_scores:
                count[vt] += 1
                for k, v in scene_scores[vt].items():
                    avgs[vt].setdefault(k, 0)
                    avgs[vt][k] += v
    for vt in avgs:
        if count[vt] > 0:
            for k in avgs[vt]:
                avgs[vt][k] /= count[vt]

    print("| 版本 | NV | Eng | Prog | 对话 | 互动 | 冲突 | 压力 | 张力 | 关系 | 目标 | 角色 |")
    print("|------|----|-----|------|------|------|------|------|------|------|------|------|")
    for vt in ["A", "B", "C"]:
        if count[vt] > 0:
            s = avgs[vt]
            print(f"| {vt} | {s['narrative_value']:.2f} | {s['engagement']:.2f} | {s['progression']:.2f} | "
                  f"{s['dialogue']:.1f} | {s['interaction']:.1f} | {s['conflict']:.1f} | {s['pressure']:.1f} | "
                  f"{s['tension']:.1f} | {s['relationship']:.1f} | {s['goal']:.1f} | {s['character']:.1f} |")


def main():
    parser = argparse.ArgumentParser(description="评分三版本")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="导出目录路径（包含 scene_*.json）")
    group.add_argument("--novel", help="小说ID（直接从数据库读取）")
    parser.add_argument("--volume", type=int, default=1, help="卷号（与--novel配合）")
    parser.add_argument("--chapter", type=int, default=1, help="章号（与--novel配合）")

    args = parser.parse_args()

    if args.input:
        score_from_dir(args.input)
    else:
        asyncio.run(score_from_db(args.novel, args.volume, args.chapter))


if __name__ == "__main__":
    main()