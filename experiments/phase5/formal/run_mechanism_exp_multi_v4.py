#!/usr/bin/env python3
# experiments/phase5/formal/run_mechanism_exp_multi_v4.py
"""
Master Pair 2 & 3: Location Gap 复制验证 (v4 最终方案)
强制 Writer 以指定的第一句开头，绕过默认行为
"""

import asyncio
import json
import sys
import re
from pathlib import Path
from datetime import datetime
import httpx

LLM_API_BASE = "http://localhost:8082/v1"
LLM_MODEL = "Qwen3-32B-Q5_K_M"


# ============================================================
# MP2: 探索→调查（古井刻字 → 回房对照典籍）
# ============================================================

MP2_SCENE_A_FIXED = """古井的青苔在月光下泛着幽绿，林逸用手电筒光束劈开浓雾时，井口溢出的寒气正凝结在他睫毛上。铁链锈迹斑驳处泛着铜绿，他拨开青苔，发现铁链上密布的环状凹痕——像是被无数细绳勒出的伤疤。

手电筒扫过井壁时，光斑在某处突然扭曲。潮湿的岩面上，用朱砂刻着三圈螺旋纹，最深处嵌着半枚指纹。他掏出匕首刮开苔藓，取出一张空白符纸，将符文拓印下来。

符纸上的纹路在月光下泛着幽蓝微光，与他在宗门古籍中见过的某种封印标记极为相似。远处传来守夜人梆子声，三更已过。"""

MP2_GAP_HIGH = """他将符纸拓印后收起匕首，站起身快步离开古井。月光下，三条岔路在他面前展开：左侧通向书房，右侧通往丹房，正中则是禁地方向。"""

MP2_GAP_LOW = """他将符纸拓印后收起匕首，沿着月光下的石径快步走向书房。夜风从身后灌入密道，吹得他衣摆猎猎作响。"""

# 强制第一句（作为生成的开头，而非背景）
MP2_FIRST_SENTENCE_HIGH = "林逸在岔路口停了下来，感受着三个方向吹来的不同风息。片刻后，他选定了通往书房的道路，迈步向前。"
MP2_FIRST_SENTENCE_LOW = "林逸快步穿过最后一道月洞门，衣袖拂过门框上的青苔，书房方向的烛火在远处晃动。"

MP2_SCENE_B = """书房，深夜。

烛火在铜灯盏里跳动，将林逸的影子投在满壁书卷上。他抽出《宗门禁制录》摊在案头，符纸压在书页边缘，螺旋纹与纸上的朱砂标记缓缓重合。窗外传来夜鸟啼叫，他将油灯拨亮，指尖沿着纹路描摹，却发现其中三道弧线与禁地外围的封印结构完全一致。

石案上的茶汤已经凉透。林逸按住符纸边缘，烛泪滴落时，他在纸背发现一行极淡的墨迹——像是被水渍浸泡过的旧字。"""


# ============================================================
# MP3: 战斗→休整（被偷袭受伤 → 石屋处理伤势并检查线索）
# ============================================================

MP3_SCENE_A_FIXED = """幽谷深处弥漫着妖兽的腥臭，林逸的玄铁剑插在岩缝里，剑刃上还沾着暗红血迹。三日前那场遭遇让他左臂多出三道深可见骨的爪痕，此刻肿胀处泛着诡异的青黑。他撕下衣摆包扎伤口，脚下踩碎了半截断裂的妖兽骨刺。月光从岩顶裂隙漏下，照见他手中那枚刚夺来的青铜令牌。

令牌表面刻着模糊的狼首纹，边角有被火焰灼烧的焦痕。他将令牌翻面，背面隐约可见一串被刮去的刻字，只剩最后一笔还能辨认——像是个"禁"字。"""

MP3_GAP_HIGH = """远处传来瀑布轰鸣声，他扶着岩壁站起身。岔路口在他面前展开：一道指向石屋方向，一道通向河谷，另一道没入密林深处。"""

MP3_GAP_LOW = """远处传来瀑布轰鸣声，他扶着岩壁站起身，朝山谷外的石屋方向走去。左臂的爪痕在夜风中阵阵刺痛。"""

MP3_FIRST_SENTENCE_HIGH = "林逸站在岔路口，能听见三条路上传来的不同声音。他紧了紧肩头的伤口，迈步走向通往石屋的那条路。"
MP3_FIRST_SENTENCE_LOW = "林逸加快脚步穿过最后一段山路，石屋的轮廓在月光下渐渐清晰。他推开虚掩的木门时，扬起的灰尘在月光中翻涌。"

MP3_SCENE_B = """石屋，深夜。

铜灯盏里剩了半盏残油，火光在窗缝漏进的冷风中摇晃。林逸卸下外袍，左臂的爪痕已经肿胀得发亮。他翻出药箱里的金疮药，倒在伤口上时，整条手臂都在抽搐。

石案上摊着那枚青铜令牌，狼首纹在烛光下泛着暗绿色锈迹。他将令牌翻到背面，被刮去的刻字在斜射的光线下，竟显出极浅的凹陷——像是有什么人用刀尖反复划过。他抓过炭笔，在纸上拓印那串凹陷，笔画断续地拼出一个字：禁。

冷风从门缝灌入，铜灯里的火苗猛地一歪。林逸按住拓纸边缘，发现凹陷深处还藏着更细的纹路——那是某种古老禁制的残影。"""


# ============================================================
# 实验执行
# ============================================================

def build_prompt(condition, first_sentence, scene_b_full, master_pair):
    """构建 Writer Prompt - 强制以 first_sentence 开头"""
    match_high = "MatchHigh" in condition
    
    if master_pair == "MP2":
        entry_location = "古井"
        scene_b_location = "书房"
        current_goal = "破译符纸上的螺旋纹含义"
    else:  # MP3
        entry_location = "幽谷"
        scene_b_location = "石屋"
        current_goal = "查明青铜令牌的来历"
    
    lines = []
    
    # ====== 关键：强制第一句 ======
    lines.append("【⚠️ 强制要求】")
    lines.append(f"你的回答必须以这句话开头：")
    lines.append(f"「{first_sentence}」")
    lines.append("")
    
    # 状态信息
    if match_high:
        lines.append("【状态信息】")
        lines.append(f"上一场景结束时你位于：{entry_location}")
        lines.append(f"当前场景发生在：{scene_b_location}")
    else:
        lines.append("【状态信息】")
        lines.append(f"当前目标：{current_goal}")
    
    lines.append("")
    
    # 承接线索和状态
    lines.append("【承接线索和状态】")
    if master_pair == "MP2":
        lines.append("请承接拓印的符纸和身上的痕迹。")
    else:
        lines.append("请承接青铜令牌和身上的伤势。")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append(scene_b_full)
    
    return "\n".join(lines)


async def call_llm(prompt: str) -> str:
    """调用 LLM 生成文本"""
    async with httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(600.0)) as client:
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 2048,
        }
        resp = await client.post(f"{LLM_API_BASE}/chat/completions", json=payload)
        data = resp.json()
        content = data["choices"][0]["message"].get("content", "")
        if not content:
            content = data["choices"][0]["message"].get("reasoning_content", "")
        return content


async def run_single(label, first_sentence, scene_b_full, condition, rep, output_dir):
    """运行单次生成 - 强制第一句"""
    prompt = build_prompt(condition, first_sentence, scene_b_full, label)
    response = await call_llm(prompt)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{label}_{condition}_rep{rep:02d}_{timestamp}.txt"
    filepath = output_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"Label: {label}\n")
        f.write(f"Condition: {condition}\n")
        f.write(f"Repeat: {rep}\n")
        f.write(f"Gap: {'High' if 'High' in condition else 'Low'}\n")
        f.write(f"Match: {'High' if 'MatchHigh' in condition else 'Low'}\n")
        f.write(f"{'='*60}\n\n")
        f.write(response)
    
    return {"label": label, "condition": condition, "rep": rep, "filepath": str(filepath)}


async def main():
    base_dir = Path(__file__).parent
    output_dir = base_dir / "reports" / "raw_location_replication_v4"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Location Gap 复制验证 (v4 最终方案: 强制第一句): MP2 + MP3")
    print("="*80)
    
    conditions = [
        "GapHigh_MatchHigh",
        "GapHigh_MatchLow",
        "GapLow_MatchHigh",
        "GapLow_MatchLow",
    ]
    
    # MP2 第一句映射
    mp2_first = {
        "GapHigh": MP2_FIRST_SENTENCE_HIGH,
        "GapLow": MP2_FIRST_SENTENCE_LOW,
    }
    
    # MP3 第一句映射
    mp3_first = {
        "GapHigh": MP3_FIRST_SENTENCE_HIGH,
        "GapLow": MP3_FIRST_SENTENCE_LOW,
    }
    
    all_results = []
    
    # MP2
    print("\n--- MP2: 探索→调查 ---")
    for cond in conditions:
        gap = "High" if "High" in cond else "Low"
        first_sentence = mp2_first[f"Gap{gap}"]
        for rep in range(3):
            print(f"  {cond} rep {rep+1}/3...")
            result = await run_single("MP2", first_sentence, MP2_SCENE_B, cond, rep, output_dir)
            all_results.append(result)
    
    # MP3
    print("\n--- MP3: 战斗→休整 ---")
    for cond in conditions:
        gap = "High" if "High" in cond else "Low"
        first_sentence = mp3_first[f"Gap{gap}"]
        for rep in range(3):
            print(f"  {cond} rep {rep+1}/3...")
            result = await run_single("MP3", first_sentence, MP3_SCENE_B, cond, rep, output_dir)
            all_results.append(result)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_runs": len(all_results),
        "master_pairs": ["MP2", "MP3"],
        "conditions": conditions,
        "repeats": 3,
        "output_dir": str(output_dir),
        "prompt_version": "v4_forced_first_sentence",
    }
    with open(output_dir.parent / "summary_location_replication_v4.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 完成! 共 {len(all_results)} 次生成")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())