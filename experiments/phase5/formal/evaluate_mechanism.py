#!/usr/bin/env python3
# experiments/phase5/formal/evaluate_mechanism.py
"""
三层评估 + Continuity
对 Master Pair 1 的 12 个样本进行评估
"""

import asyncio
import json
import re
from pathlib import Path
import httpx
from collections import defaultdict
import statistics

LLM_API_BASE = "http://localhost:8082/v1"
LLM_MODEL = "Qwen3-32B-Q5_K_M"

SCENE_A = """炼丹房的青砖地面残留着焦黑的爪痕，空气中浮动着血瞳虎独有的硫磺味。林逸单膝跪地，左肩的伤口正往外渗血，袖口被利爪撕成布条。他低头看了眼怀中半卷《太乙化神录》，虎爪扯断秘银锁链时发出的脆响还在耳边回响。

虎爪最后一击擦过他耳际时，他分明感觉到风压割裂了皮肤。此刻肩胛骨的灼痛反倒让他清醒过来——这种伤至少得用冰魄兰调敷三日。

他扶着墙根站直，血珠顺着指缝滴落在焦痕上，发出细微的嗞嗞声。远处传来巡逻弟子的脚步声。"""

SCENE_A_HIGH_GAP = "他提起卷轴，快步离开炼丹房。身后传来石门合拢的闷响，密道在他面前分出三条岔路：左侧通往静室方向，右侧通向丹房，正中则是长老堂的暗道。林逸没有犹豫太久，选择了其中一条。"

SCENE_A_LOW_GAP = "他提起卷轴，拖着受伤的左肩，朝通向静室的密道走去。石壁上的烛火在穿堂风中明灭，他数着自己的脚步声，直到静室的门在昏暗光线下显出轮廓。"

SCENE_B = """静室，深夜。

烛火在铜灯盏里跳动，将林逸的影子投在粗糙的石壁上。他卸下外袍，露出左肩撕裂的伤口——虎爪留下的四道血痕边缘泛着不自然的青黑，在火光中格外狰狞。

石案上摊着《太乙化神录》的半卷残页，羊皮纸边缘被血渍浸透，几处朱砂标记在烛光下泛着暗红。他扯过绷带裹住肩头，动作扯动伤口时，冷汗顺着下颌滑落。随即他将注意力转回卷轴，残页上的星图与验灵阵纹路重叠，三处被涂改的标记正指向同一个位置——禁地西北角的废弃丹房。

冷风从气窗钻入，火苗歪斜，石案上的药瓶轻轻晃动。林逸按住卷轴边缘，指尖在涂改处反复摩挲，仿佛那些朱砂里还藏着另一层信息。"""

# 条件定义
CONDITIONS = {
    "A1_GapHigh_MatchHigh": {"gap": "high", "match": "high", "label": "A"},
    "B1_GapHigh_MatchLow": {"gap": "high", "match": "low", "label": "B"},
    "C1_GapLow_MatchHigh": {"gap": "low", "match": "high", "label": "C"},
    "D1_GapLow_MatchLow": {"gap": "low", "match": "low", "label": "D"},
}

def load_texts(raw_dir: Path) -> list:
    """加载所有生成的文本"""
    texts = []
    for f in sorted(raw_dir.glob("*.txt")):
        with open(f, 'r') as file:
            content = file.read()
        match = re.search(r'={60}\n\n(.*?)\n\n={60}', content, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            text = content.strip()
        
        # 提取 label
        label = f.stem.split('_')[0] + '_' + f.stem.split('_')[1]
        texts.append({
            "filename": f.name,
            "label": label,
            "text": text,
            "full_content": content
        })
    return texts

async def evaluate_sample(scene_b_text: str, state_injected: str, debug: bool = False) -> dict:
    """对单个样本进行三层评估"""
    
    prompt = f"""你是叙事分析专家。请分析以下场景文本，评估林逸的状态信息是否被识别、整合和利用。

【场景文本】
{scene_b_text[:800]}

【注入的状态信息】
{state_injected}

请从以下四个维度评分（1-5）：

1. Recognition（识别）：文本中是否明确提到了注入的状态信息？
   - 5：明确提及，措辞清晰
   - 3：隐约暗示
   - 1：完全没有出现

2. Integration（整合）：状态信息是否被写入正文并成为场景的一部分？
   - 5：深度融入叙事，与场景自然交织
   - 3：被提及但孤立
   - 1：完全没有被整合

3. Utilization（利用）：状态信息是否影响了角色的决策或情节走向？
   - 5：成为决策的关键因素，影响剧情
   - 3：有轻微影响
   - 1：完全没有影响

4. Continuity（连续性）：场景之间的整体连续性如何？
   - 5：无缝衔接，逻辑连贯
   - 3：基本连贯但有轻微断裂
   - 1：明显断裂

输出 JSON 格式：
{{
    "recognition": 1-5,
    "integration": 1-5,
    "utilization": 1-5,
    "continuity": 1-5,
    "reason": "简短理由"
}}"""

    async with httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(300.0)) as client:
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 300,
        }
        resp = await client.post(f"{LLM_API_BASE}/chat/completions", json=payload)
        data = resp.json()
        content = data["choices"][0]["message"].get("content", "")
        if not content:
            content = data["choices"][0]["message"].get("reasoning_content", "")
        
        if debug:
            print(f"  [DEBUG] 内容长度: {len(content)}")
            print(f"  [DEBUG] 内容预览: {content[:150]}...")
        
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        
        # 降级：提取数字
        numbers = re.findall(r'[1-5]', content)
        if len(numbers) >= 4:
            return {
                "recognition": int(numbers[0]),
                "integration": int(numbers[1]),
                "utilization": int(numbers[2]),
                "continuity": int(numbers[3]),
                "reason": "从文本提取数字"
            }
        
        return {
            "recognition": 3,
            "integration": 3,
            "utilization": 3,
            "continuity": 3,
            "reason": "解析失败"
        }

async def main():
    base_dir = Path(__file__).parent
    raw_dir = base_dir / "reports" / "raw_m1"
    output_dir = base_dir / "reports" / "m1_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("机制评估: Master Pair 1 (Location Gap)")
    print("="*80)
    
    texts = load_texts(raw_dir)
    print(f"找到 {len(texts)} 个样本")
    
    # 按条件分组
    by_condition = defaultdict(list)
    for item in texts:
        label = item["label"]
        if label in CONDITIONS:
            by_condition[label].append(item)
    
    # 定义各条件的注入状态
    INJECTION_STATES = {
        "A1_GapHigh_MatchHigh": "上一场景结束时位于炼丹房，当前场景发生在静室，需要从炼丹房自然过渡到静室。",
        "B1_GapHigh_MatchLow": "当前目标是找出《太乙化神录》中的异常标记。",
        "C1_GapLow_MatchHigh": "上一场景结束时位于炼丹房，当前场景发生在静室，需要从炼丹房自然过渡到静室。",
        "D1_GapLow_MatchLow": "当前目标是找出《太乙化神录》中的异常标记。",
    }
    
    all_results = []
    
    for label, items in by_condition.items():
        print(f"\n--- {label} ---")
        print(f"  Gap: {CONDITIONS[label]['gap']}, Match: {CONDITIONS[label]['match']}")
        
        state_injected = INJECTION_STATES.get(label, "")
        
        for item in items:
            print(f"  评估: {item['filename']}...")
            result = await evaluate_sample(item["text"], state_injected, debug=False)
            result["label"] = label
            result["filename"] = item["filename"]
            all_results.append(result)
            print(f"    R:{result['recognition']} I:{result['integration']} U:{result['utilization']} C:{result['continuity']}")
    
    # 统计
    print("\n" + "="*80)
    print("统计结果")
    print("="*80)
    
    # 按条件汇总
    for label in ["A1_GapHigh_MatchHigh", "B1_GapHigh_MatchLow", "C1_GapLow_MatchHigh", "D1_GapLow_MatchLow"]:
        items = [r for r in all_results if r["label"] == label]
        if items:
            n = len(items)
            r = statistics.mean([i["recognition"] for i in items])
            integ = statistics.mean([i["integration"] for i in items])
            u = statistics.mean([i["utilization"] for i in items])
            c = statistics.mean([i["continuity"] for i in items])
            print(f"\n{label}:")
            print(f"  Recognition: {r:.2f} (n={n})")
            print(f"  Integration: {integ:.2f}")
            print(f"  Utilization: {u:.2f}")
            print(f"  Continuity: {c:.2f}")
    
    # 保存结果
    output_path = output_dir / "m1_scores.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())