#!/usr/bin/env python
"""
Phase 15.1 — Rewrite Experiment (修正版)

- 3 个真实场景
- Original / Minimal / Constrained
- Minimal 不加任何结构化指令
- Constrained 从真实 contract 提取约束
- 输出组织为便于盲读的结构
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import init_db_pool, close_db_pool, get_db_pool
from openai import AsyncOpenAI
import httpx

NOVEL_ID = "simple_long_novel_001"
OUTPUT_DIR = Path("experiments/phase15_1_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 3 个场景：从真实生产数据中选取
SCENES = [
    {"volume": 3, "chapter": 90, "scene_idx": 1},
    {"volume": 3, "chapter": 90, "scene_idx": 2},
    {"volume": 1, "chapter": 8, "scene_idx": 0},
]

# ============================================================
# 真正的 Minimal Prompt（不加任何预置指令）
# ============================================================
MINIMAL_PROMPT = """
下面是一段已经生成的小说正文。

请在不改变原文所发生事情的前提下，把它润色得更自然、更像正式小说。

不要添加解释。直接输出修改后的正文。

原文：
{original_text}
"""

# ============================================================
# Constrained Prompt（约束从真实数据提取）
# ============================================================
CONSTRAINED_PROMPT_TEMPLATE = """
下面是一段已经生成的小说正文。

【必须保留的事件】
{events}

【必须保留的角色】
{characters}

【禁止新增】
- 不要增加新的剧情主线
- 不要增加新的世界事实（如"三年前曾来过这里"）
- 不要增加新角色
- 不要改变人物关系

你可以自由发挥的方面：
- 环境描写、感官细节
- 人物动作和身体反应
- 心理活动
- 对话之间的停顿、表情、动作
- 句子节奏和段落划分

你的任务不是重写故事，而是让读者"亲身经历"已经发生的这些事件。

直接输出改写后的正文，不要添加解释。

原文：
{original_text}
"""

# ============================================================
# 辅助函数
# ============================================================

async def fetch_scene_text(volume, chapter, scene_idx):
    """从 narrative_versions 获取原始场景正文"""
    pool = get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT scene_text
            FROM narrative_versions
            WHERE novel_id = $1
              AND volume_num = $2
              AND chapter_num = $3
              AND scene_idx = $4
              AND version_type = 'A'
            ORDER BY generated_at DESC
            LIMIT 1
            """,
            NOVEL_ID, volume, chapter, scene_idx
        )
        if not row:
            return None
        raw = row["scene_text"]
        try:
            data = json.loads(raw)
            return data.get("scene_text", raw)
        except:
            return raw

async def extract_contract_info(volume, chapter, scene_idx):
    """从 scene_execution_units 提取真实的 contract 信息"""
    pool = get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT plan_json
            FROM scene_execution_units
            WHERE novel_id = $1
              AND volume_num = $2
              AND chapter_num = $3
              AND scene_index = $4
            """,
            NOVEL_ID, volume, chapter, scene_idx
        )
        if not row:
            return [], []
        plan = json.loads(row["plan_json"])
        contract = plan.get("planning_contract", {})
        observables = contract.get("observables", {})
        state_changes = observables.get("state_changes", [])
        events = []
        for sc in state_changes:
            sc_type = sc.get("type", "")
            if sc_type == "plot_flag":
                events.append(f"{sc.get('name')} 发生")
            elif sc_type == "location_change":
                events.append(f"{sc.get('actor')} 到达 {sc.get('location')}")
            elif sc_type == "relationship_change":
                events.append(f"{sc.get('from_char')} 与 {sc.get('to_char')} 关系变化 {sc.get('delta')}")
            elif sc_type == "knowledge_gain":
                events.append(f"获得知识：{sc.get('name')}")
            elif sc_type == "inventory_acquire":
                events.append(f"{sc.get('actor')} 获得 {sc.get('item')}")
        characters = plan.get("characters", [])
        return events, characters

async def call_llm(prompt):
    """调用 LLM"""
    client = AsyncOpenAI(
        api_key="not-needed",
        base_url="http://localhost:8082",
        timeout=httpx.Timeout(120.0),
    )
    response = await client.chat.completions.create(
        model="Qwen3-32B-Q5_K_M",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=4096,
    )
    return response.choices[0].message.content or ""

# ============================================================
# 主实验
# ============================================================

async def main():
    await init_db_pool()
    print("Phase 15.1 — Rewrite Experiment (修正版)")
    print("=" * 60)

    results = {}

    for scene in SCENES:
        vol = scene["volume"]
        ch = scene["chapter"]
        idx = scene["scene_idx"]
        scene_id = f"scene_{vol}_{ch}_{idx}"

        print(f"\n处理场景: {scene_id}")

        original = await fetch_scene_text(vol, ch, idx)
        if not original:
            print(f"  ❌ 未找到原文")
            continue

        print(f"  ✅ 原文长度: {len(original)}")

        # 提取真实约束
        events, characters = await extract_contract_info(vol, ch, idx)
        if not events:
            events = ["（从 contract 中未提取到具体事件，请人工判断）"]
        if not characters:
            characters = ["（从 contract 中未提取到角色，请人工判断）"]

        events_text = "\n".join(f"- {e}" for e in events)
        characters_text = ", ".join(characters) if characters else "（未知）"

        print(f"  📋 提取到 {len(events)} 个事件, {len(characters)} 个角色")

        # Minimal
        min_prompt = MINIMAL_PROMPT.format(original_text=original)
        print("  🔄 Minimal Rewrite...")
        min_result = await call_llm(min_prompt)

        # Constrained
        constrained_prompt = CONSTRAINED_PROMPT_TEMPLATE.format(
            original_text=original,
            events=events_text,
            characters=characters_text,
        )
        print("  🔄 Constrained Rewrite...")
        constrained_result = await call_llm(constrained_prompt)

        # 保存
        scene_dir = OUTPUT_DIR / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        with open(scene_dir / "original.txt", "w", encoding="utf-8") as f:
            f.write(original)
        with open(scene_dir / "minimal.txt", "w", encoding="utf-8") as f:
            f.write(min_result)
        with open(scene_dir / "constrained.txt", "w", encoding="utf-8") as f:
            f.write(constrained_result)

        # 保存元数据供人工参考
        with open(scene_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({
                "scene_id": scene_id,
                "volume": vol,
                "chapter": ch,
                "scene_idx": idx,
                "extracted_events": events,
                "extracted_characters": characters,
                "original_length": len(original),
                "minimal_length": len(min_result),
                "constrained_length": len(constrained_result),
            }, f, indent=2, ensure_ascii=False)

        results[scene_id] = {
            "original_length": len(original),
            "minimal_length": len(min_result),
            "constrained_length": len(constrained_result),
        }

        print(f"  ✅ 已保存到 {scene_dir}")

    print("\n" + "=" * 60)
    print("实验完成")
    print("=" * 60)

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n结果保存在: {OUTPUT_DIR}/")
    print("\n请按以下顺序阅读（盲读建议）：")
    print("1. 先不打开 original，只看 minimal 和 constrained")
    print("2. 判断哪个更像小说")
    print("3. 再打开 original 对比，判断是否改变了故事")
    print("\n每个场景目录:")
    for scene_id in results:
        print(f"  - {scene_id}/")

    await close_db_pool()

if __name__ == "__main__":
    asyncio.run(main())