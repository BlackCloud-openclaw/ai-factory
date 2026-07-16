#!/usr/bin/env python3
"""
Blind Ranking + Score Comparison
同时输出：二选一结果 + 两个文本的 1-5 分
"""

import asyncio
import json
import re
from pathlib import Path
import httpx

LLM_API_BASE = "http://localhost:8082/v1"
LLM_MODEL = "Qwen3-32B-Q5_K_M"

SCENE_A_TEXTS = {
    "pair_01": """炼丹房的青砖地面残留着焦黑的爪痕，空气中浮动着血瞳虎独有的硫磺味。林逸单膝跪地，左肩的伤口正往外渗血，袖口被利爪撕成布条。他低头看了眼怀中半卷《太乙化神录》，虎爪扯断秘银锁链时发出的脆响还在耳边回响。

虎爪最后一击擦过他耳际时，他分明感觉到风压割裂了皮肤。此刻肩胛骨的灼痛反倒让他清醒过来——这种伤至少得用冰魄兰调敷三日。

他扶着墙根站直，血珠顺着指缝滴落在焦痕上，发出细微的嗞嗞声。远处传来巡逻弟子的脚步声，他提起卷轴，拖着受伤的左肩，朝通向静室的密道走去。""",

    "pair_10": """清晨的药园弥漫着露水与药草的气息，林逸提着竹篮穿梭在灵草间。他弯腰采下一株紫纹参时，指尖忽然触到异样的腥甜——本该清冽的药香里，混着一丝腐烂果核的腐臭。

他拨开半人高的玉灵草，发现三株血灵芝的叶片正蜷曲发黑，根部渗出暗褐色黏液。远处传来守药人的脚步声，他将血灵芝连根挖起，快步离开药园。"""
}


def load_text(raw_dir: Path, pattern: str) -> str:
    files = sorted(raw_dir.glob(pattern))
    if not files:
        return ""
    with open(files[0], 'r') as f:
        content = f.read()
    match = re.search(r'={60}\n\n(.*?)\n\n={60}', content, re.DOTALL)
    return match.group(1).strip() if match else content[:500]


async def call_judge(scene_a: str, text_a: str, text_b: str, debug: bool = False) -> dict:
    prompt = f"""你是叙事连续性评估专家。

【场景 A】
{scene_a}

【场景 B1】
{text_a}

【场景 B2】
{text_b}

请完成两个任务：
1. 选择：B1 和 B2 哪一个更自然地承接场景 A？（只输出 B1 或 B2）
2. 评分：分别给 B1 和 B2 的连续性打分（1-5，5为最高）

输出 JSON：
{{
    "choice": "B1" 或 "B2",
    "score_B1": 1-5,
    "score_B2": 1-5,
    "reason": "简短理由"
}}"""

    async with httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(300.0)) as client:
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1024,
        }
        resp = await client.post(f"{LLM_API_BASE}/chat/completions", json=payload)
        data = resp.json()
        content = data["choices"][0]["message"].get("content", "")
        if not content:
            content = data["choices"][0]["message"].get("reasoning_content", "")

        if debug:
            print(f"[DEBUG] 内容长度: {len(content)}")
            print(f"[DEBUG] 内容预览: {content[:200]}...")

        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass

        # 如果解析失败，尝试从文本中提取数字
        numbers = re.findall(r'[1-5]', content)
        if len(numbers) >= 4:
            return {
                "choice": "B2" if "B2" in content else "B1",
                "score_B1": int(numbers[0]),
                "score_B2": int(numbers[1]),
                "reason": "从数字提取"
            }
        return {"choice": "ERROR", "score_B1": 0, "score_B2": 0, "reason": "解析失败"}


async def main():
    raw_dir = Path(__file__).parent / "reports" / "raw"
    test_cases = [
        ("pair_01", "C2"),
        ("pair_10", "C2"),
    ]

    print("="*80)
    print("Blind Ranking + Score Comparison")
    print("="*80)

    for pair, cond in test_cases:
        print(f"\n--- {pair} {cond} ---")
        scene_a = SCENE_A_TEXTS.get(pair)
        if not scene_a:
            print("  警告: 无 Scene A")
            continue

        text_base = load_text(raw_dir, f"{pair}_baseline_rep00_*.txt")
        text_cond = load_text(raw_dir, f"{pair}_{cond}_rep00_*.txt")
        if not text_base or not text_cond:
            print("  警告: 无文本")
            continue

        print(f"  Baseline 长度: {len(text_base)}, {cond} 长度: {len(text_cond)}")

        result = await call_judge(scene_a, text_base, text_cond, debug=True)
        print(f"  Choice: {result.get('choice')}")
        print(f"  Score_B1 (Baseline): {result.get('score_B1')}")
        print(f"  Score_B2 ({cond}): {result.get('score_B2')}")
        print(f"  Reason: {result.get('reason', '')[:100]}")

        # 检查一致性
        choice = result.get("choice")
        score_b1 = result.get("score_B1")
        score_b2 = result.get("score_B2")
        if choice == "B2" and score_b2 < score_b1:
            print("  ⚠️ 不一致: 选择了 B2 但 B2 的分数低于 B1")
        elif choice == "B1" and score_b1 < score_b2:
            print("  ⚠️ 不一致: 选择了 B1 但 B1 的分数低于 B2")
        else:
            print("  ✅ 一致: 选择与评分相符")


if __name__ == "__main__":
    asyncio.run(main())