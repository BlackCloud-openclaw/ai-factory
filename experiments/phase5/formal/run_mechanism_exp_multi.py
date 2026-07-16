#!/usr/bin/env python3
# experiments/phase5/formal/run_mechanism_exp_multi.py
"""
Master Pair 2 & 3: Location Gap 复制验证
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

MP2_GAP_HIGH = """他将符纸拓印后收起匕首，站起身快步离开古井。密道在月光下分出三条路：左侧通向书房，右侧通往丹房，正中则是禁地方向。夜风从三个方向灌入，裹着不同的气息——书墨、药草、铁锈。林逸没有迟疑太久，转身走向其中一条。"""

MP2_GAP_LOW = """他将符纸拓印后收起匕首，快步走回书房。石壁上的烛火在穿堂风中明灭，他数着自己的脚步声，直到书房的木门在昏暗中显出轮廓。符纸上的螺旋纹在袖中隐隐发烫。"""

MP2_SCENE_B = """书房，深夜。

烛火在铜灯盏里跳动，将林逸的影子投在满壁书卷上。他抽出《宗门禁制录》摊在案头，符纸压在书页边缘，螺旋纹与纸上的朱砂标记缓缓重合。窗外传来夜鸟啼叫，他将油灯拨亮，指尖沿着纹路描摹，却发现其中三道弧线与禁地外围的封印结构完全一致。

石案上的茶汤已经凉透。林逸按住符纸边缘，烛泪滴落时，他在纸背发现一行极淡的墨迹——像是被水渍浸泡过的旧字。"""


# ============================================================
# MP3: 战斗→休整（被偷袭受伤 → 石屋处理伤势并检查线索）
# ============================================================

MP3_SCENE_A_FIXED = """幽谷深处弥漫着妖兽的腥臭，林逸的玄铁剑插在岩缝里，剑刃上还沾着暗红血迹。三日前那场遭遇让他左臂多出三道深可见骨的爪痕，此刻肿胀处泛着诡异的青黑。他撕下衣摆包扎伤口，脚下踩碎了半截断裂的妖兽骨刺。月光从岩顶裂隙漏下，照见他手中那枚刚夺来的青铜令牌。

令牌表面刻着模糊的狼首纹，边角有被火焰灼烧的焦痕。他将令牌翻面，背面隐约可见一串被刮去的刻字，只剩最后一笔还能辨认——像是个"禁"字。"""

MP3_GAP_HIGH = """远处传来瀑布轰鸣声，他扶着岩壁站起身。月光在岔路口投下交错树影，一道指向石屋方向，一道通向河谷，另一道没入密林深处。他收起令牌，朝其中一条走去。"""

MP3_GAP_LOW = """远处传来瀑布轰鸣声，他扶着岩壁站起身，朝山谷外的石屋方向走去。左臂的爪痕在夜风中刺痛，他加快脚步，石屋的轮廓在月光下渐渐清晰。"""

MP3_SCENE_B = """石屋，深夜。

铜灯盏里剩了半盏残油，火光在窗缝漏进的冷风中摇晃。林逸卸下外袍，左臂的爪痕已经肿胀得发亮。他翻出药箱里的金疮药，倒在伤口上时，整条手臂都在抽搐。

石案上摊着那枚青铜令牌，狼首纹在烛光下泛着暗绿色锈迹。他将令牌翻到背面，被刮去的刻字在斜射的光线下，竟显出极浅的凹陷——像是有什么人用刀尖反复划过。他抓过炭笔，在纸上拓印那串凹陷，笔画断续地拼出一个字：禁。

冷风从门缝灌入，铜灯里的火苗猛地一歪。林逸按住拓纸边缘，发现凹陷深处还藏着更细的纹路——那是某种古老禁制的残影。"""


# ============================================================
# 实验执行
# ============================================================

def build_prompt(condition, scene_a_full, scene_b_full):
    """构建完整的 Writer Prompt"""
    gap_high = "High" in condition
    match_high = "MatchHigh" in condition
    
    if match_high:
        prefix = """【场景起点】
上一场景结束时你位于：{entry_location}
当前场景发生在：{scene_b_location}
请从上一场景自然过渡到当前场景，并承接所有线索和状态。

"""
    else:
        prefix = """【场景起点】
当前目标：{current_goal}
请从上一场景自然过渡到当前场景，并承接所有线索和状态。

"""
    
    if "MP2" in condition:
        entry_location = "古井"
        scene_b_location = "书房"
        current_goal = "破译符纸上的螺旋纹含义"
    else:  # MP3
        entry_location = "幽谷"
        scene_b_location = "石屋"
        current_goal = "查明青铜令牌的来历"
    
    if match_high:
        prefix = prefix.format(entry_location=entry_location, scene_b_location=scene_b_location)
    else:
        prefix = prefix.format(current_goal=current_goal)
    
    return prefix + "\n---\n\n" + scene_b_full


async def call_llm(prompt: str) -> str:
    """调用 LLM 生成文本"""
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


async def run_single(label, scene_a_full, scene_b_full, condition, rep, output_dir):
    """运行单次生成"""
    prompt = build_prompt(condition, scene_a_full, scene_b_full)
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
    output_dir = base_dir / "reports" / "raw_location_replication"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Location Gap 复制验证: MP2 + MP3")
    print("="*80)
    
    # 定义所有条件
    conditions = [
        "GapHigh_MatchHigh",
        "GapHigh_MatchLow",
        "GapLow_MatchHigh",
        "GapLow_MatchLow",
    ]
    
    # MP2 配置
    mp2_scenes = {
        "GapHigh": MP2_GAP_HIGH,
        "GapLow": MP2_GAP_LOW,
    }
    mp2_scene_b = MP2_SCENE_B
    
    # MP3 配置
    mp3_scenes = {
        "GapHigh": MP3_GAP_HIGH,
        "GapLow": MP3_GAP_LOW,
    }
    mp3_scene_b = MP3_SCENE_B
    
    all_results = []
    
    # 运行 MP2
    print("\n--- MP2: 探索→调查 ---")
    for cond in conditions:
        gap = "High" if "High" in cond else "Low"
        scene_a_full = MP2_SCENE_A_FIXED + "\n\n" + mp2_scenes[f"Gap{gap}"]
        label = "MP2"
        for rep in range(3):
            print(f"  {cond} rep {rep+1}/3...")
            result = await run_single(label, scene_a_full, mp2_scene_b, cond, rep, output_dir)
            all_results.append(result)
    
    # 运行 MP3
    print("\n--- MP3: 战斗→休整 ---")
    for cond in conditions:
        gap = "High" if "High" in cond else "Low"
        scene_a_full = MP3_SCENE_A_FIXED + "\n\n" + mp3_scenes[f"Gap{gap}"]
        label = "MP3"
        for rep in range(3):
            print(f"  {cond} rep {rep+1}/3...")
            result = await run_single(label, scene_a_full, mp3_scene_b, cond, rep, output_dir)
            all_results.append(result)
    
    # 汇总
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_runs": len(all_results),
        "master_pairs": ["MP2", "MP3"],
        "conditions": conditions,
        "repeats": 3,
        "output_dir": str(output_dir),
    }
    with open(output_dir.parent / "summary_location_replication.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 完成! 共 {len(all_results)} 次生成")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())