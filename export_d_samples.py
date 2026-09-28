import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from dotenv import load_dotenv
load_dotenv(project_root / ".env")
import asyncio
import json
from src.db import init_db_pool, get_db_pool

async def main():
    await init_db_pool()
    pool = get_db_pool()
    if not pool:
        print("No pool")
        return
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT 
                id,
                scene_id,
                original_text,
                rewritten_text,
                original_passed,
                rewritten_passed,
                status,
                contract_data,
                writer_events,
                original_violations,
                rewritten_violations,
                executed_at
            FROM shadow_rewrite_log
            WHERE original_passed = false 
              AND rewritten_passed = false
              AND status = 'validation_failed'
              AND original_text IS NOT NULL
            ORDER BY executed_at DESC
            LIMIT 30
        """)
        if not rows:
            print("No samples found")
            return
        samples = []
        for r in rows:
            samples.append({
                "id": r["id"],
                "scene_id": r["scene_id"],
                "original_text": r["original_text"],
                "rewritten_text": r["rewritten_text"],
                "original_passed": r["original_passed"],
                "rewritten_passed": r["rewritten_passed"],
                "status": r["status"],
                "contract_data": r["contract_data"],
                "writer_events": r["writer_events"],
                "original_violations": r["original_violations"],
                "rewritten_violations": r["rewritten_violations"],
                "executed_at": str(r["executed_at"]) if r["executed_at"] else None,
            })
        output_path = project_root / "d_quadrant_30_samples.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"✅ 导出 {len(samples)} 个样本到 {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
