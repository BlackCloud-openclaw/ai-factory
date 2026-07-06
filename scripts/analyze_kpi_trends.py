#!/usr/bin/env python
"""
全量 KPI 趋势分析 - Phase 5 实测
自动计算第 1-20 章的 Narrative KPI，追踪戏剧性演变
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.writing.narrative_kpi import NarrativeKPIEngine


def main():
    engine = NarrativeKPIEngine()
    data = []

    print("📊 AI Factory 叙事价值趋势分析 (第 1-20 章)")
    print("=" * 80)

    for i in range(1, 21):
        # 读取章节
        fpath = Path(f"data/novels/simple_long_novel_001/vol_001/chap_{i:03d}.txt")
        if not fpath.exists():
            print(f"⚠️ 第 {i} 章文件不存在，跳过")
            continue

        text = fpath.read_text(encoding="utf-8")

        # 计算 KPI（使用空状态差异，仅基于文本特征）
        # 注意：Relationship/Goal/Character 可能因缺少 state_diff 而偏低，
        # 但相对趋势是有效的
        result = engine.compute(text, {}, {})

        data.append({
            "chapter": i,
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
            "chars": result.total_chars,
        })

        print(f"第 {i:02d} 章 | NV: {result.narrative_value:.2f} | "
              f"对话: {result.dialogue:.1f} | 冲突: {result.conflict:.1f} | "
              f"张力: {result.tension:.1f} | 角色: {result.character:.1f}")

    print("=" * 80)

    # 输出完整 Markdown 表格
    print("\n## 📋 完整数据表\n")
    print("| 章 | NV | Eng | Prog | 对话 | 互动 | 冲突 | 压力 | 张力 | 关系 | 目标 | 角色 |")
    print("|----|----|-----|------|------|------|------|------|------|------|------|------|")
    for d in data:
        print(f"| {d['chapter']:02d} | {d['narrative_value']:.2f} | {d['engagement']:.2f} | {d['progression']:.2f} | "
              f"{d['dialogue']:.1f} | {d['interaction']:.1f} | {d['conflict']:.1f} | {d['pressure']:.1f} | "
              f"{d['tension']:.1f} | {d['relationship']:.1f} | {d['goal']:.1f} | {d['character']:.1f} |")

    # 分析关键段落
    if len(data) >= 18:
        print("\n## 📈 关键观测\n")
        nv_16 = data[15]["narrative_value"]   # 第16章
        nv_18 = data[17]["narrative_value"]   # 第18章
        nv_20 = data[19]["narrative_value"]   # 第20章
        print(f"- 第16章（探索/机关）: NV = {nv_16:.2f}")
        print(f"- 第18章（剑阁/认知反转）: NV = {nv_18:.2f}")
        print(f"- 第20章（异象/隐患/战斗）: NV = {nv_20:.2f}")
        print(f"- 增量: +{nv_18 - nv_16:.2f} (第16→18章), +{nv_20 - nv_18:.2f} (第18→20章)")

    # 判断是否存在 Director 介入信号
    if len(data) >= 18 and nv_18 > nv_16 + 0.5:
        print("\n✅ 检测到戏剧性跃升（第16→18章），与 Director 介入点位一致。")
    else:
        print("\n⚠️ 未检测到显著戏剧性跃升，需进一步排查。")

if __name__ == "__main__":
    main()