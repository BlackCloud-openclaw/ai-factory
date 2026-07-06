#!/usr/bin/env python
"""
A/B/C 三版本 KPI 对比工具
用法: python scripts/compare_versions_kpi.py --novel simple_long_novel_001 [--chapter 16] [--output report.md]
默认输出 Markdown 格式到终端。
"""

import asyncio
import asyncpg
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import config


async def fetch_versions(novel_id: str, chapter: int = None):
    """从数据库获取版本数据"""
    conn = await asyncpg.connect(config.postgres_dsn)
    try:
        query = """
            SELECT volume_num, chapter_num, scene_idx, version_type, kpi_scores
            FROM narrative_versions
            WHERE novel_id = $1
        """
        params = [novel_id]
        if chapter is not None:
            query += " AND chapter_num = $2"
            params.append(chapter)
        query += " ORDER BY chapter_num, scene_idx, version_type"
        rows = await conn.fetch(query, *params)
        return rows
    finally:
        await conn.close()


def parse_kpi(kpi_json: str) -> Dict[str, float]:
    """解析 KPI JSON 字段"""
    if not kpi_json:
        return {}
    try:
        return json.loads(kpi_json)
    except:
        return {}


def compute_stats(scores: List[float]) -> Dict[str, float]:
    """计算均值、标准差、最小值、最大值"""
    if not scores:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    n = len(scores)
    mean = sum(scores) / n
    variance = sum((x - mean) ** 2 for x in scores) / n
    return {
        "mean": mean,
        "std": variance ** 0.5,
        "min": min(scores),
        "max": max(scores),
        "count": n,
    }


def generate_report(rows, novel_id: str, chapter: int = None):
    """生成 Markdown 报告"""
    # 按章节和场景聚合
    chapters_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # chapters_data[chapter_num][scene_idx][version_type] = list of KPI dicts (每个场景只有一个版本，但这里为通用设计)

    for row in rows:
        vol = row["volume_num"]
        ch = row["chapter_num"]
        scene = row["scene_idx"]
        vtype = row["version_type"]
        kpi = parse_kpi(row["kpi_scores"])
        if kpi:
            chapters_data[ch][scene][vtype] = kpi

    # 收集所有章节
    chapters = sorted(chapters_data.keys())
    if not chapters:
        print("未找到任何版本数据。")
        return

    lines = []
    lines.append(f"# A/B/C 版本 KPI 对比报告")
    lines.append(f"**小说**: {novel_id}")
    if chapter:
        lines.append(f"**指定章节**: 第 {chapter} 章")
    else:
        lines.append(f"**章节范围**: 第 {chapters[0]} ~ {chapters[-1]} 章")
    lines.append(f"**总场景数**: {sum(len(scenes) for scenes in chapters_data.values())}")
    lines.append("")

    # 维度列表（从第一个KPI中提取）
    first_kpi = None
    for ch_data in chapters_data.values():
        for scene_data in ch_data.values():
            for vtype, kpi in scene_data.items():
                if kpi:
                    first_kpi = kpi
                    break
            if first_kpi:
                break
        if first_kpi:
            break

    if not first_kpi:
        lines.append("❌ 无法解析任何 KPI 数据。")
        print("\n".join(lines))
        return

    dims = list(first_kpi.keys())
    # 移除 'versions' 等元数据字段
    dims = [d for d in dims if d not in ("versions", "total_chars", "total_events")]

    # 按版本汇总统计数据
    version_stats = {v: {d: {"values": []} for d in dims} for v in ["A", "B", "C"]}
    version_counts = defaultdict(int)

    for ch, scenes in chapters_data.items():
        for scene, versions in scenes.items():
            for vtype, kpi in versions.items():
                if vtype in version_stats:
                    version_counts[vtype] += 1
                    for d in dims:
                        if d in kpi:
                            version_stats[vtype][d]["values"].append(kpi[d])

    # 计算统计
    version_summary = {}
    for vtype in ["A", "B", "C"]:
        version_summary[vtype] = {}
        for d in dims:
            vals = version_stats[vtype][d]["values"]
            if vals:
                version_summary[vtype][d] = compute_stats(vals)
            else:
                version_summary[vtype][d] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}

    # 输出表格
    lines.append("## 各版本均值对比")
    header = "| 版本 | " + " | ".join(dims) + " | 场景数 |"
    sep = "|" + " --- |" * (len(dims) + 2)
    lines.append(header)
    lines.append(sep)
    for vtype in ["A", "B", "C"]:
        row = f"| **{vtype}** "
        for d in dims:
            mean = version_summary[vtype][d]["mean"]
            row += f"| {mean:.2f} "
        row += f"| {version_counts[vtype]} |"
        lines.append(row)

    # 计算 B-A 和 C-B 的增量
    lines.append("")
    lines.append("## B vs A 增量（戏剧结构增量）")
    row = "| 维度 "
    for d in dims:
        row += f"| {d} "
    row += "|"
    lines.append(row)
    sep = "|" + " --- |" * (len(dims) + 1)
    lines.append(sep)
    inc_row = "| **B - A** "
    for d in dims:
        delta = version_summary["B"][d]["mean"] - version_summary["A"][d]["mean"]
        inc_row += f"| {delta:+.2f} "
    inc_row += "|"
    lines.append(inc_row)

    lines.append("")
    lines.append("## C vs B 增量（润色增量）")
    inc_row = "| **C - B** "
    for d in dims:
        delta = version_summary["C"][d]["mean"] - version_summary["B"][d]["mean"]
        inc_row += f"| {delta:+.2f} "
    inc_row += "|"
    lines.append(inc_row)

    # 额外：总体 Narrative Value 的增量
    lines.append("")
    lines.append("## 关键指标：Narrative Value")
    nv_means = {}
    for vtype in ["A", "B", "C"]:
        nv_means[vtype] = version_summary[vtype]["narrative_value"]["mean"]
    lines.append(f"- A 平均 NV: {nv_means['A']:.3f}")
    lines.append(f"- B 平均 NV: {nv_means['B']:.3f}  (Δ = {nv_means['B'] - nv_means['A']:+.3f})")
    lines.append(f"- C 平均 NV: {nv_means['C']:.3f}  (Δ = {nv_means['C'] - nv_means['B']:+.3f})")

    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description="A/B/C KPI 对比工具")
    parser.add_argument("--novel", default="simple_long_novel_001", help="小说 ID")
    parser.add_argument("--chapter", type=int, help="指定章节（可选）")
    parser.add_argument("--output", help="输出文件路径（可选）")
    args = parser.parse_args()

    rows = await fetch_versions(args.novel, args.chapter)
    report = generate_report(rows, args.novel, args.chapter)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"报告已保存至 {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    asyncio.run(main())