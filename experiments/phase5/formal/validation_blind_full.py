#!/usr/bin/env python3
# experiments/phase5/formal/validation_blind_full.py
"""
完整 Blind Ranking：所有 Pair × 所有 Intervention
不告诉 Judge 实验条件，只问：哪个更连续？
"""

import asyncio
import json
import re
from pathlib import Path
from collections import defaultdict
import httpx

LLM_API_BASE = "http://localhost:8082/v1"
LLM_MODEL = "Qwen3-32B-Q5_K_M"

SCENE_A_TEXTS = {
    "pair_01": """炼丹房的青砖地面残留着焦黑的爪痕，空气中浮动着血瞳虎独有的硫磺味。林逸单膝跪地，左肩的伤口正往外渗血，袖口被利爪撕成布条。他低头看了眼怀中半卷《太乙化神录》，虎爪扯断秘银锁链时发出的脆响还在耳边回响。

虎爪最后一击擦过他耳际时，他分明感觉到风压割裂了皮肤。此刻肩胛骨的灼痛反倒让他清醒过来——这种伤至少得用冰魄兰调敷三日。

他扶着墙根站直，血珠顺着指缝滴落在焦痕上，发出细微的嗞嗞声。远处传来巡逻弟子的脚步声，他提起卷轴，拖着受伤的左肩，朝通向静室的密道走去。""",
    "pair_02": """幽谷深处弥漫着妖兽的腥臭，林逸的玄铁剑插在岩缝里，剑刃上还沾着暗红血迹。三日前那场遭遇让他左臂多出三道深可见骨的爪痕，此刻肿胀处泛着诡异的青黑。

他撕下衣摆包扎伤口，脚下踩碎了半截断裂的妖兽骨刺。月光从岩顶裂隙漏下，照见他手中那枚刚夺来的青铜令牌——上面刻着的狼首纹与宗门通缉令上的标记如出一辙。

远处传来瀑布轰鸣声，他扶着岩壁站起身，朝山谷外的石屋方向走去。""",
    "pair_03": """藏书阁第三层的书架积着半寸厚的灰，林逸推开《云笈七签》时，夹层里滑出半张泛黄的拓片。星图暗纹在月光下泛起幽蓝光晕，他立刻认出这与验灵阵第三重纹路完全重合。

他展开拓片，边缘的撕裂处还残留着朱砂印记。这些符号与他在禁地外围见过的标记如出一辙。窗外传来夜巡弟子经过的脚步声，他将拓片卷起塞入衣襟内侧，合上书册，快步走向阁楼的侧梯。""",
    "pair_04": """古井的青苔在月光下泛着幽绿，林逸用手电筒光束劈开浓雾时，井口溢出的寒气正凝结在他睫毛上。铁链锈迹斑驳处泛着铜绿，他拨开青苔，发现铁链上密布的环状凹痕——像是被无数细绳勒出的伤疤。

手电筒扫过井壁时，光斑在某处突然扭曲。潮湿的岩面上，用朱砂刻着三圈螺旋纹，最深处嵌着半枚指纹。他掏出匕首刮开更多苔藓，取出怀中的拓印纸，将符文拓下后快步离开古井。""",
    "pair_05": """议事堂的檀木桌被拍得震响，管事的茶杯盖子跳了两跳。林逸直视着对方泛红的脖颈，指节在袖中握得发白。

「灵田分配的事，什么时候轮到杂役弟子开口？」管事的唾沫星子溅在账册上。

林逸松开拳头，声音比预想中平静：「那就让长老亲自验灵田的灵气浓度。」

他转身踏出议事堂门槛时，晚风裹着青草气扑面而来。身后传来杯盏碎裂的声响，他没有回头，沿着石阶朝山道走去。""",
    "pair_06": """擂台上散落的碎石还带着血迹，林逸单膝跪地，右手按着被玄冰剑划伤的腰侧。台下弟子们的议论声嗡嗡作响，莫玄收剑入鞘时嘴角那抹冷笑像根刺扎进他眼底。

「外门弟子也敢挑战内门？不自量力。」莫玄丢下这句话时，袖口扫起的尘灰扑了林逸一脸。

林逸撑着剑柄站起身，没有回话。他扯断束发的布条，任由汗湿的长发遮住半边脸，转身走下擂台，朝山脚下的河边走去。""",
    "pair_07": """地牢的霉味混着铁锈气息扑面而来，林逸蹲在囚笼前，用匕首挑起地上半截烧焦的符纸。审讯台上的鞭痕还渗着暗红，俘虏的目光从低垂的乱发间透出来，嘴唇翕动着吐出半句残音：「...深渊入口在...」

话音未落，俘虏突然浑身抽搐，嘴角溢出黑血。林逸冲上前撬开他的嘴，却只摸到半片咬碎的毒囊。他握紧符纸站起身，朝地牢外的密道快步走去。""",
    "pair_08": """灵田的稻穗在夜风中沙沙作响，林逸蹲在田埂上，指尖抚过一片被踩断的稻秆。断口处还残留着淡淡的灵气余韵，三道平行的压痕延伸向东北方的密林深处。

他掏出怀中的罗盘，指针疯狂旋转后锁定东北方。守夜人的火把在远处移动，他将罗盘收入衣襟，顺着压痕的方向快步追去。""",
    "pair_09": """林逸盘膝坐在静室蒲团上，掌心托着半块玉简，丹田处灵力如潮水般涌动。他运转《御气诀》冲击炼气三层时，忽然察觉心神一阵恍惚——眼前浮现出七岁那年被逐出师门的画面，宗门长老的铜锁砸碎肩骨的声音仿佛就在耳边。

他猛地咬破舌尖，血珠溅在玉简上激起青光：「修炼时心魔作祟，不能停！」""",
    "pair_10": """清晨的药园弥漫着露水与药草的气息，林逸提着竹篮穿梭在灵草间。他弯腰采下一株紫纹参时，指尖忽然触到异样的腥甜——本该清冽的药香里，混着一丝腐烂果核的腐臭。

他拨开半人高的玉灵草，发现三株血灵芝的叶片正蜷曲发黑，根部渗出暗褐色黏液。远处传来守药人的脚步声，他将血灵芝连根挖起，快步离开药园。"""
}


def load_texts(raw_dir: Path, pattern: str) -> list:
    """加载匹配 pattern 的文本文件"""
    texts = []
    for f in sorted(raw_dir.glob(pattern)):
        with open(f, 'r') as file:
            content = file.read()
        match = re.search(r'={60}\n\n(.*?)\n\n={60}', content, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            text = content[:500]
        texts.append(text)
    return texts


async def call_judge(scene_a: str, text_a: str, text_b: str, debug: bool = False) -> str:
    """调用 Judge，返回选择 B1 或 B2"""
    prompt = f"""请阅读以下三段场景，判断【场景 B1】和【场景 B2】哪一个更自然地承接【场景 A】的位置和状态。

【场景 A】
{scene_a}

【场景 B1】
{text_a}

【场景 B2】
{text_b}

请只输出一个选项：B1 或 B2。
不要解释，不要评分，只输出选项。"""

    async with httpx.AsyncClient(
        trust_env=False,
        timeout=httpx.Timeout(300.0, connect=30.0)
    ) as client:
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 20,
        }
        try:
            response = await client.post(
                f"{LLM_API_BASE}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"].get("content", "")
            if not content:
                content = data["choices"][0]["message"].get("reasoning_content", "")
            
            if debug:
                print(f"  [DEBUG] Judge 返回: {content[:50]}...")
            
            content_upper = content.strip().upper()
            if "B1" in content_upper and "B2" not in content_upper:
                return "B1"
            elif "B2" in content_upper and "B1" not in content_upper:
                return "B2"
            else:
                numbers = re.findall(r'[12]', content)
                if numbers:
                    return f"B{numbers[-1]}"
                return "无法判断"
                
        except Exception as e:
            print(f"  [ERROR] {e}")
            return "ERROR"


async def main():
    base_dir = Path(__file__).parent
    raw_dir = base_dir / "reports" / "raw"
    
    print("="*80)
    print("完整 Blind Ranking：所有 Pair × 所有 Intervention")
    print("="*80)
    
    pairs = [f"pair_{i:02d}" for i in range(1, 11)]
    interventions = ["C1", "C2", "C3", "C4"]
    
    all_results = defaultdict(lambda: defaultdict(dict))
    
    for pair in pairs:
        print(f"\n--- {pair} ---")
        scene_a = SCENE_A_TEXTS.get(pair)
        if not scene_a:
            print(f"  警告: 找不到 {pair} 的 Scene A")
            continue
        
        baseline_texts = load_texts(raw_dir, f"{pair}_baseline_rep*.txt")
        if not baseline_texts:
            print(f"  警告: 找不到 {pair} 的 Baseline 文本")
            continue
        
        for cond in interventions:
            cond_texts = load_texts(raw_dir, f"{pair}_{cond}_rep*.txt")
            if not cond_texts:
                print(f"  警告: 找不到 {pair} {cond} 的文本")
                continue
            
            # 每个干预选第一个文本 vs 第一个 Baseline
            text_base = baseline_texts[0]
            text_cond = cond_texts[0]
            
            print(f"  {cond}: Baseline vs {cond}")
            print(f"    Baseline 长度: {len(text_base)}, {cond} 长度: {len(text_cond)}")
            
            result = await call_judge(scene_a, text_base, text_cond, debug=True)
            print(f"    Judge 选择: {result}")
            
            all_results[pair][cond] = {
                "winner": result,
                "baseline_len": len(text_base),
                "cond_len": len(text_cond)
            }
    
    # 统计
    print("\n" + "="*80)
    print("统计结果：Intervention 被选为更连续的次数")
    print("="*80)
    
    cond_wins = {c: 0 for c in interventions}
    cond_losses = {c: 0 for c in interventions}
    cond_undecided = {c: 0 for c in interventions}
    
    for pair, conds in all_results.items():
        for cond, result in conds.items():
            if result["winner"] == "B2":
                cond_wins[cond] += 1
            elif result["winner"] == "B1":
                cond_losses[cond] += 1
            else:
                cond_undecided[cond] += 1
    
    print(f"\n{'Intervention':<12} {'Wins':<8} {'Losses':<8} {'Undecided':<10} {'Win Rate':<10}")
    for cond in interventions:
        total = cond_wins[cond] + cond_losses[cond] + cond_undecided[cond]
        win_rate = cond_wins[cond] / total if total > 0 else 0
        print(f"{cond:<12} {cond_wins[cond]:<8} {cond_losses[cond]:<8} {cond_undecided[cond]:<10} {win_rate:.0%}")
    
    # 保存结果
    output_path = base_dir / "reports" / "validation_blind_full.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())