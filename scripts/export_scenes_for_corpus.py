#!/usr/bin/env python3
"""
从 narrative_versions 表导出场景文本，用于手工构建 Gold Corpus
"""

import os
import json
import asyncio
import asyncpg
from pathlib import Path

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", 5432))
DB_NAME = os.environ.get("DB_NAME", "ai_factory")
DB_USER = os.environ.get("DB_USER", "woami")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "kali")


async def export_scenes(novel_id: str, limit: int = 30, output_dir: Path = Path("experiments/phase12/corpus/candidates")):
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = await asyncpg.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
    )

    try:
        rows = await conn.fetch("""
            SELECT 
                id,
                volume_num,
                chapter_num,
                scene_idx,
                version_type,
                scene_text
            FROM narrative_versions
            WHERE novel_id = $1
                AND scene_text IS NOT NULL
                AND LENGTH(scene_text) > 200
            ORDER BY id DESC
            LIMIT $2
        """, novel_id, limit)

        print(f"Found {len(rows)} candidate scenes\n")
        print("=" * 70)

        for i, row in enumerate(rows, 1):
            try:
                scene_data = json.loads(row["scene_text"])
                scene_text = scene_data.get("scene_text", str(row["scene_text"]))
            except:
                scene_text = str(row["scene_text"])

            print(f"[{i}] Volume {row['volume_num']}, Chapter {row['chapter_num']}, Scene {row['scene_idx']} (v{row['version_type']})")
            print("-" * 40)
            print(f"Text: {scene_text[:300]}..." if len(scene_text) > 300 else f"Text: {scene_text}")
            print("=" * 70)
            print()

            candidate_file = output_dir / f"candidate_{i:03d}.json"
            with open(candidate_file, "w", encoding="utf-8") as f:
                json.dump({
                    "id": row["id"],
                    "volume": row["volume_num"],
                    "chapter": row["chapter_num"],
                    "scene_idx": row["scene_idx"],
                    "version": row["version_type"],
                    "scene_text": scene_text,
                }, f, indent=2, ensure_ascii=False)

        print(f"\n✅ {len(rows)} candidates saved to {output_dir}")

    finally:
        await conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--novel", default="simple_long_novel_001")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", default="experiments/phase12/corpus/candidates")
    args = parser.parse_args()

    asyncio.run(export_scenes(
        novel_id=args.novel,
        limit=args.limit,
        output_dir=Path(args.output),
    ))