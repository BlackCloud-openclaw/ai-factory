#!/usr/bin/env python3
# experiments/phase5/formal/run_mechanism_exp.py

import asyncio
import json
import re
from pathlib import Path
from datetime import datetime
import httpx

LLM_API_BASE = "http://localhost:8082/v1"
LLM_MODEL = "Qwen3-32B-Q5_K_M"

# ============================================================
# Master Pair 1: Location Gap
# ============================================================

SCENE_A_BASE = """炼丹房的青砖地面残留着焦黑的爪痕，空气中浮动着血瞳虎独有的硫磺味。林逸单膝跪地，左肩的伤口正往外渗血，袖口被利爪撕成布条。他低头看了眼怀中半卷《太乙化神录》，虎爪扯断秘银锁链时发出的脆响还在耳边回响。

虎爪最后一击擦过他耳际时，他分明感觉到风压割裂了皮肤。此刻肩胛骨的灼痛反倒让他清醒过来——这种伤至少得用冰魄兰调敷三日。

他扶着墙根站直，血珠顺着指缝滴落在焦痕上，发出细微的嗞嗞声。远处传来巡逻弟子的脚步声。"""

SCENE_A_HIGH_GAP = "他提起卷轴，快步离开炼丹房。身后传来石门合拢的闷响，密道在他面前分出三条岔路：左侧通往静室方向，右侧通向丹房，正中则是长老堂的暗道。林逸没有犹豫太久，选择了其中一条。"

SCENE_A_LOW_GAP = "他提起卷轴，拖着受伤的左肩，朝通向静室的密道走去。石壁上的烛火在穿堂风中明灭，他数着自己的脚步声，直到静室的门在昏暗光线下显出轮廓。"

SCENE_B = """烛火在铜灯盏里跳动，将林逸的影子投在粗糙的石壁上。他卸下外袍，露出左肩撕裂的伤口——虎爪留下的四道血痕边缘泛着不自然的青黑，在火光中格外狰狞。

石案上摊着《太乙化神录》的半卷残页，羊皮纸边缘被血渍浸透，几处朱砂标记在烛光下泛着暗红。他扯过绷带裹住肩头，动作扯动伤口时，冷汗顺着下颌滑落。随即他将注意力转回卷轴，残页上的星图与验灵阵纹路重叠，三处被涂改的标记正指向同一个位置——禁地西北角的废弃丹房。

冷风从气窗钻入，火苗歪斜，石案上的药瓶轻轻晃动。林逸按住卷轴边缘，指尖在涂改处反复摩挲，仿佛那些朱砂里还藏着另一层信息。"""


def build_prompt(gap_high: bool, match_high: bool, repeat: int) -> str:
    # 构建场景 A 文本
    scene_a_full = SCENE_A_BASE + "\n\n" + (SCENE_A_HIGH_GAP if gap_high else SCENE_A_LOW_GAP)
    
    # 状态注入
    if match_high:
        state_injection = "【场景起点】\n上一场景结束时你位于：炼丹房\n当前场景发生在：静室\n请从炼丹房自然过渡到静室，并承接林逸身上的伤势和手中的卷轴。"
    else:
        state_injection = "【场景起点】\n当前目标：找出《太乙化神录》中的异常标记\n请从上一场景自然过渡，并承接林逸身上的伤势和手中的卷轴。"
    
    prompt = f"""请根据以下场景计划和状态信息，生成一段场景正文（约 400-600 字）。

【场景 A（前置）】
{scene_a_full}

【场景 B（目标场景）】
{SCENE_B}

{state_injection}

【写作要求】
1. 自然承接场景 A 的位置和状态。
2. 描写林逸进入静室后的行动，包括处理伤口和检查卷轴。
3. 保持语言风格一致，使用第三人称。
4. 只输出场景正文，不要包含任何额外说明或元信息。

请直接输出场景正文（纯文本），不要使用 JSON 或其他格式。"""

    return prompt


async def call_llm(prompt: str) -> str:
    async with httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(600.0)) as client:
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 2048,
        }
        resp = await client.post(f"{LLM_API_BASE}/chat/completions", json=payload)
        data = resp.json()
        content = data["choices"][0]["message"].get("content", "")
        if not content:
            content = data["choices"][0]["message"].get("reasoning_content", "")
        return content


async def main():
    base_dir = Path(__file__).parent
    output_dir = base_dir / "reports" / "raw_m1"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Master Pair 1: Location Gap 实验 (直接调用)")
    print("="*60)

    conditions = [
        ("A1_GapHigh_MatchHigh", True, True),
        ("B1_GapHigh_MatchLow", True, False),
        ("C1_GapLow_MatchHigh", False, True),
        ("D1_GapLow_MatchLow", False, False),
    ]

    for label, gap_high, match_high in conditions:
        print(f"\n--- {label} ---")
        print(f"  Gap High: {gap_high}, Match High: {match_high}")

        for rep in range(3):
            prompt = build_prompt(gap_high, match_high, rep)
            print(f"  生成 rep {rep+1}/3...")
            text = await call_llm(prompt)

            # 保存
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{label}_rep{rep:02d}_{timestamp}.txt"
            filepath = output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Label: {label}\n")
                f.write(f"Repeat: {rep}\n")
                f.write(f"Gap: {'High' if gap_high else 'Low'}, Match: {'High' if match_high else 'Low'}\n")
                f.write(f"{'='*60}\n\n")
                f.write(text)

    print(f"\n✅ 完成! 12 次生成")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())