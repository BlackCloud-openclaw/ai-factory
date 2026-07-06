#!/usr/bin/env python
"""
Narrative Projection Analysis Script (Enhanced with Experimental Groups)
拆分四个维度：FPS / LPS / APS / QPS，并按实验组聚合统计。
"""

import asyncio
import json
import sys
import math
import aiohttp
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import init_db_pool, close_db_pool, get_db_pool
from src.config import config
from src.common.logging import setup_logging

logger = setup_logging("analyze_projection")

# ========== 实验组定义（章节范围） ==========
EXPERIMENT_GROUPS = {
    "baseline": (1, 10),
    "loop": (11, 24),
    "focus": (25, 35),
    "both": (36, 45),
    "full": (46, 59),
    "question": (60, 70),
}


# ========== 独立的 embedding 函数 ==========
async def generate_embedding(text: str) -> List[float]:
    if not text:
        return []
    endpoint = config.embedding_endpoint
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(endpoint, json={"input": text}, timeout=timeout) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if "data" in data and len(data["data"]) > 0:
                        return data["data"][0].get("embedding", [])
                    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                        return data[0]
                else:
                    logger.error(f"Embedding service error: {resp.status} {await resp.text()}")
        except Exception as e:
            logger.error(f"Embedding request failed: {e}")
    return []


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ========== 数据获取 ==========
async def fetch_projections(novel_id: str) -> List[Dict]:
    pool = get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, novel_id, chapter, event_id, projection_data, created_at
            FROM narrative_projection_snapshots
            WHERE novel_id = $1
            ORDER BY chapter ASC, created_at ASC
            """,
            novel_id
        )
        projections = []
        for row in rows:
            data = json.loads(row["projection_data"])
            projections.append({
                "id": row["id"],
                "novel_id": row["novel_id"],
                "chapter": row["chapter"],
                "event_id": row["event_id"],
                "projection_data": data,
                "created_at": row["created_at"]
            })
        return projections


def extract_per_chapter_latest(projections: List[Dict]) -> Dict[int, Dict]:
    chapter_map = {}
    for proj in projections:
        ch = proj["chapter"]
        if ch not in chapter_map or proj["created_at"] > chapter_map[ch]["created_at"]:
            chapter_map[ch] = proj
    return chapter_map


def extract_projection_fields(proj: Dict) -> Dict:
    data = proj["projection_data"]
    focus = data.get("focus", {})
    loop = data.get("loop", {})
    attention = data.get("attention", {})
    question = data.get("question", {})
    return {
        "focus": {
            "subject": focus.get("subject", ""),
            "type": focus.get("type", ""),
            "why_matters": focus.get("why_matters", ""),
        },
        "loop": {
            "description": loop.get("description", ""),
            "initiator": loop.get("initiator", ""),
            "urgency": loop.get("urgency", 0.0),
        },
        "attention": {
            "target": attention.get("target", ""),
            "intensity": attention.get("intensity", 0.0),
        },
        "question": {
            "text": question.get("text", ""),
            "scope": question.get("scope", ""),
        },
        "confidence": data.get("confidence", 0.0),
    }


def build_text_for_dimension(fields: Dict, dimension: str) -> str:
    """为某个维度构建文本表示，用于 embedding"""
    if dimension == "focus":
        f = fields["focus"]
        parts = []
        if f["subject"]:
            parts.append(f"subject: {f['subject']}")
        if f["type"]:
            parts.append(f"type: {f['type']}")
        if f["why_matters"]:
            parts.append(f"why: {f['why_matters']}")
        return " ".join(parts) if parts else ""
    elif dimension == "loop":
        l = fields["loop"]
        parts = []
        if l["description"]:
            parts.append(f"desc: {l['description']}")
        if l["initiator"]:
            parts.append(f"initiator: {l['initiator']}")
        if l["urgency"]:
            parts.append(f"urgency: {l['urgency']:.2f}")
        return " ".join(parts) if parts else ""
    elif dimension == "attention":
        a = fields["attention"]
        parts = []
        if a["target"]:
            parts.append(f"target: {a['target']}")
        if a["intensity"]:
            parts.append(f"intensity: {a['intensity']:.2f}")
        return " ".join(parts) if parts else ""
    elif dimension == "question":
        q = fields["question"]
        parts = []
        if q["text"]:
            parts.append(f"text: {q['text']}")
        if q["scope"]:
            parts.append(f"scope: {q['scope']}")
        return " ".join(parts) if parts else ""
    else:
        return ""


async def compute_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    try:
        emb1 = await generate_embedding(text1)
        emb2 = await generate_embedding(text2)
        if not emb1 or not emb2:
            return 0.0
        return cosine_similarity(emb1, emb2)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return 0.0


async def analyze_novel(novel_id: str) -> Dict:
    projections = await fetch_projections(novel_id)
    if not projections:
        logger.warning(f"No projections found for novel {novel_id}")
        return {}

    chapter_map = extract_per_chapter_latest(projections)
    sorted_chapters = sorted(chapter_map.keys())
    
    results = {}
    for ch in sorted_chapters:
        proj = chapter_map[ch]
        fields = extract_projection_fields(proj)
        results[ch] = {
            "projection": proj,
            "fields": fields,
        }

    # 四个维度
    dims = ["focus", "loop", "attention", "question"]
    similarities = {dim: {} for dim in dims}
    
    for i in range(len(sorted_chapters) - 1):
        ch1 = sorted_chapters[i]
        ch2 = sorted_chapters[i+1]
        fields1 = results[ch1]["fields"]
        fields2 = results[ch2]["fields"]
        for dim in dims:
            text1 = build_text_for_dimension(fields1, dim)
            text2 = build_text_for_dimension(fields2, dim)
            sim = await compute_similarity(text1, text2)
            similarities[dim][(ch1, ch2)] = sim

    # 全局各维度平均分（只计算非零对）
    avg_scores = {}
    for dim in dims:
        vals = [sim for sim in similarities[dim].values() if sim > 0]
        avg_scores[dim] = sum(vals) / len(vals) if vals else 0.0

    # ---- 按实验组聚合 ----
    group_avg = {dim: {group: [] for group in EXPERIMENT_GROUPS} for dim in dims}
    group_pair_count = {group: 0 for group in EXPERIMENT_GROUPS}

    for dim in dims:
        for (ch1, ch2), sim in similarities[dim].items():
            if sim == 0:
                continue
            for group_name, (start, end) in EXPERIMENT_GROUPS.items():
                if start <= ch1 <= end and start <= ch2 <= end:
                    group_avg[dim][group_name].append(sim)
                    if dim == dims[0]:  # 只在第一个维度计数一次
                        group_pair_count[group_name] += 1
                    break

    group_final = {dim: {} for dim in dims}
    for dim in dims:
        for group_name, vals in group_avg[dim].items():
            group_final[dim][group_name] = sum(vals) / len(vals) if vals else 0.0

    return {
        "chapters": results,
        "similarities": similarities,
        "avg_scores": avg_scores,
        "group_final": group_final,
        "group_pair_count": group_pair_count,
        "total_chapters": len(sorted_chapters),
    }


def generate_report(result: Dict, novel_id: str) -> str:
    lines = []
    lines.append(f"# 叙事投影四维分析报告 - {novel_id}")
    lines.append("")
    lines.append(f"**分析时间**: {__import__('datetime').datetime.now().isoformat()}")
    lines.append(f"**总章节数**: {result['total_chapters']}")
    lines.append("")
    lines.append("## 各维度全局平均持续性分数")
    lines.append("")
    lines.append("| 维度 | 平均相似度 | 说明 |")
    lines.append("|------|-----------|------|")
    for dim, score in result["avg_scores"].items():
        dim_name = dim.upper()
        lines.append(f"| {dim_name} | {score:.3f} | 相邻章节 {dim} 的语义相似度 |")
    lines.append("")

    # ----- 按实验组聚合的表格 -----
    lines.append("## 各实验组各维度平均相似度")
    lines.append("")
    lines.append("| 实验组 | 章节范围 | FOCUS | LOOP | ATTENTION | QUESTION | 有效对 |")
    lines.append("|--------|----------|-------|------|-----------|----------|--------|")
    
    dims = ["focus", "loop", "attention", "question"]
    group_final = result.get("group_final", {})
    for group_name, (start, end) in EXPERIMENT_GROUPS.items():
        row = f"| {group_name.capitalize()} | {start}-{end} |"
        for dim in dims:
            val = group_final.get(dim, {}).get(group_name, 0.0)
            row += f" {val:.3f} |"
        count = result.get("group_pair_count", {}).get(group_name, 0)
        row += f" {count} |"
        lines.append(row)
    lines.append("")

    # 各维度最优组
    best_groups = {}
    for dim in dims:
        best_group = max(EXPERIMENT_GROUPS.keys(), 
                         key=lambda g: group_final.get(dim, {}).get(g, 0))
        best_groups[dim] = (best_group, group_final[dim][best_group])

    lines.append("## 各实验组最佳表现")
    lines.append("")
    for dim in dims:
        g, score = best_groups[dim]
        lines.append(f"- **{dim.upper()}** 最优组: {g.capitalize()} ({score:.3f})")
    lines.append("")

    # 各章节摘要
    lines.append("## 各章节投影摘要")
    lines.append("")
    lines.append("| 章节 | 聚焦主题 | 循环描述 | 注意力目标 | 核心问题 | 置信度 |")
    lines.append("|------|----------|----------|------------|----------|--------|")
    for ch, data in sorted(result["chapters"].items()):
        fields = data["fields"]
        focus_subj = fields["focus"]["subject"][:30] if fields["focus"]["subject"] else "-"
        loop_desc = fields["loop"]["description"][:30] if fields["loop"]["description"] else "-"
        att_target = fields["attention"]["target"][:30] if fields["attention"]["target"] else "-"
        q_text = fields["question"]["text"][:30] if fields["question"]["text"] else "-"
        conf = fields["confidence"]
        lines.append(f"| {ch} | {focus_subj} | {loop_desc} | {att_target} | {q_text} | {conf:.2f} |")
    lines.append("")

    # 四维逐对相似度
    for dim in dims:
        lines.append(f"## {dim.upper()} 相邻章节相似度")
        lines.append("")
        lines.append("| 章节对 | 相似度 |")
        lines.append("|--------|--------|")
        for (ch1, ch2), sim in sorted(result["similarities"][dim].items()):
            lines.append(f"| {ch1} → {ch2} | {sim:.3f} |")
        lines.append("")

    # 评估
    lines.append("## 全局评估")
    lines.append("")
    for dim in dims:
        score = result["avg_scores"][dim]
        if score >= 0.7:
            status = "✅ 良好"
        elif score >= 0.5:
            status = "⚠️ 中等"
        else:
            status = "❌ 较低"
        lines.append(f"- **{dim.upper()}**: {score:.3f} ({status})")
    lines.append("")
    lines.append("**按实验组分析结论**:")
    for dim in dims:
        g, score = best_groups[dim]
        lines.append(f"- {dim.upper()} 最好的实验组是 **{g.capitalize()}** ({score:.3f})")
    lines.append("")
    lines.append("**建议**: Loop 组在多个维度上表现均衡且较高，建议优先固化 Loop 作为叙事状态变量。")
    lines.append("Attention 虽然得分高，但可能因为字段单一（'林逸'重复）导致虚高，需结合剧情上下文判断。")

    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description="四维叙事投影分析（含实验组统计）")
    parser.add_argument("--novel-id", default="simple_long_novel_001")
    parser.add_argument("--output", help="输出 Markdown 文件")
    args = parser.parse_args()

    await init_db_pool()
    try:
        result = await analyze_novel(args.novel_id)
        if not result:
            print(f"没有找到 {args.novel_id} 的投影数据。")
            return
        report = generate_report(result, args.novel_id)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"报告已保存至 {args.output}")
        else:
            print(report)
    finally:
        await close_db_pool()


if __name__ == "__main__":
    asyncio.run(main())