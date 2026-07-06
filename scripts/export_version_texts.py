#!/usr/bin/env python
"""
导出 A/B/C 三版本场景正文文本
用法: 
  python scripts/export_version_texts.py --novel simple_long_novel_001
  python scripts/export_version_texts.py --novel simple_long_novel_001 --chapter 16
  python scripts/export_version_texts.py --novel simple_long_novel_001 --output ./exports
"""

import asyncio
import asyncpg
import json
import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import config


async def fetch_versions(novel_id: str, chapter: int = None):
    """从数据库获取版本数据（包含 scene_text）"""
    conn = await asyncpg.connect(config.postgres_dsn)
    try:
        query = """
            SELECT volume_num, chapter_num, scene_idx, version_type, scene_text
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


def extract_plain_text(scene_text_json: str) -> str:
    """
    从可能包含 JSON 包装的 scene_text 中提取纯文本正文。
    如果 scene_text 是 JSON 对象且包含 "scene_text" 键，则提取该值；
    否则假设本身就是纯文本。
    """
    if not scene_text_json:
        return ""
    # 尝试解析 JSON
    try:
        data = json.loads(scene_text_json)
        if isinstance(data, dict) and "scene_text" in data:
            # 如果提取到的 scene_text 本身也是 JSON 字符串，递归提取
            inner = data["scene_text"]
            try:
                inner_data = json.loads(inner)
                if isinstance(inner_data, dict) and "scene_text" in inner_data:
                    return inner_data["scene_text"]
                return inner
            except:
                return inner
        else:
            return scene_text_json
    except:
        # 不是 JSON，直接返回原字符串
        return scene_text_json


async def export_texts(novel_id: str, output_dir: str, chapter: int = None):
    """导出所有版本文本到指定目录"""
    rows = await fetch_versions(novel_id, chapter)
    if not rows:
        print(f"未找到任何版本数据 (novel_id={novel_id})")
        return

    base_dir = Path(output_dir) / novel_id
    base_dir.mkdir(parents=True, exist_ok=True)

    # 按章节分组
    chapters_data: Dict[int, Dict[int, Dict[str, str]]] = {}
    for row in rows:
        ch = row["chapter_num"]
        scene = row["scene_idx"]
        vtype = row["version_type"]
        text = extract_plain_text(row["scene_text"])
        chapters_data.setdefault(ch, {}).setdefault(scene, {})[vtype] = text

    total_scenes = sum(len(scenes) for scenes in chapters_data.values())
    print(f"找到 {len(chapters_data)} 章, {total_scenes} 个场景")

    # 导出每个场景
    for ch, scenes in sorted(chapters_data.items()):
        ch_dir = base_dir / f"ch_{ch:03d}"
        ch_dir.mkdir(exist_ok=True)
        for scene_idx, versions in sorted(scenes.items()):
            for vtype in ["A", "B", "C"]:
                if vtype in versions:
                    text = versions[vtype]
                    if not text.strip():
                        text = "(空内容)"
                    file_name = f"scene_{scene_idx:02d}_{vtype}.txt"
                    file_path = ch_dir / file_name
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    print(f"✅ 导出: {file_path}")
                else:
                    print(f"⚠️ 场景 {ch}-{scene_idx} 缺少 {vtype} 版本")

    # 可选：生成一个汇总文件，列出每个场景的版本信息
    summary_file = base_dir / "summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"小说: {novel_id}\n")
        f.write(f"导出时间: {asyncio.get_event_loop().time()}\n")
        f.write(f"总章节数: {len(chapters_data)}\n")
        f.write(f"总场景数: {total_scenes}\n")
        f.write("\n各场景版本存在情况:\n")
        for ch, scenes in sorted(chapters_data.items()):
            f.write(f"\n第 {ch} 章:\n")
            for scene_idx, versions in sorted(scenes.items()):
                present = [v for v in ["A", "B", "C"] if v in versions]
                f.write(f"  场景 {scene_idx:02d}: {', '.join(present)}\n")

    print(f"\n📁 所有文件已导出到: {base_dir}")
    print(f"📄 汇总文件: {summary_file}")


async def main():
    parser = argparse.ArgumentParser(description="导出 A/B/C 三版本正文文本")
    parser.add_argument("--novel", default="simple_long_novel_001", help="小说 ID")
    parser.add_argument("--chapter", type=int, help="指定章节（可选）")
    parser.add_argument("--output", default="./exports", help="输出根目录（默认 ./exports）")
    args = parser.parse_args()

    await export_texts(args.novel, args.output, args.chapter)


if __name__ == "__main__":
    asyncio.run(main())