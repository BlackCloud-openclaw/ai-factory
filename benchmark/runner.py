#!/usr/bin/env python
import asyncio
import json
import yaml
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import init_db_pool, close_db_pool, get_db_pool
from src.writing.state_loader import load_state_at_chapter
from benchmark.domain_extractors import *

async def run_benchmark_for_chapter(novel_id: str, volume: int, chapter: int):
    world, compressed = await load_state_at_chapter(novel_id, volume, chapter)
    if world is None:
        return {"error": f"无法加载第 {volume} 卷第 {chapter} 章状态"}

    queries_path = Path(__file__).parent / "queries" / "domain.yaml"
    with open(queries_path) as f:
        config = yaml.safe_load(f)

    results = {}
    for q in config["queries"]:
        extractor_name = q["extractor"]
        extractor = globals().get(extractor_name)
        if not extractor:
            results[q["id"]] = {"error": f"Extractor {extractor_name} not found"}
            continue
        args = q.get("args", [])
        try:
            value = extractor(world, compressed, *args)
        except Exception as e:
            value = f"error: {e}"
        results[q["id"]] = value
    return results

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except:
        return "unknown"

async def generate_baseline(novel_id: str):
    await init_db_pool()
    pool = get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT volume_num, chapter_num
            FROM narrative_events
            WHERE novel_id = $1
            ORDER BY volume_num, chapter_num
            """,
            novel_id
        )
    volumes_chapters = [(row["volume_num"], row["chapter_num"]) for row in rows]
    print(f"发现 {len(volumes_chapters)} 个章节")

    baseline = {
        "schema_version": 1,
        "novel_id": novel_id,
        "git_commit": get_git_commit(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator_version": "benchmark_v1",
        "chapter_count": len(volumes_chapters),
        "chapters": {}
    }

    for vol, ch in volumes_chapters:
        print(f"处理第 {vol} 卷第 {ch} 章...")
        res = await run_benchmark_for_chapter(novel_id, vol, ch)
        baseline["chapters"][f"{vol}_{ch}"] = res

    await close_db_pool()
    output_path = Path(__file__).parent / "baseline" / f"{novel_id}.json"
    with open(output_path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"基线已保存到 {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--novel-id", default="simple_long_novel_001")
    parser.add_argument("--generate-baseline", action="store_true")
    args = parser.parse_args()
    if args.generate_baseline:
        asyncio.run(generate_baseline(args.novel_id))
    else:
        print("请指定 --generate-baseline")
