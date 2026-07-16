#!/usr/bin/env python3
# experiments/phase5/formal/evaluator.py
"""
Phase 5 正式实验评估器 - 完整修复版
直接嵌入全部场景 A 文本，解决导入问题
"""

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import httpx
from dataclasses import dataclass

LLM_API_BASE = "http://localhost:8082/v1"
LLM_MODEL = "Qwen3-32B-Q5_K_M"

DIMENSIONS = [
    "spatial",
    "physical",
    "intentional",
    "informational",
    "temporal",
    "narrative_dependency"
]


@dataclass
class Sample:
    scene_id: str
    condition: str
    repeat: int
    text: str
    pair_id: str


@dataclass
class JudgeResult:
    scene_id: str
    condition: str
    repeat: int
    spatial_score: int
    physical_score: int
    intentional_score: int
    informational_score: int
    temporal_score: int
    narrative_dependency_score: int
    brief_reason: str


# ============================================================
# 全部 10 个场景 A 文本（硬编码，避免导入依赖）
# ============================================================

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


def load_samples(raw_dir: Path) -> List[Sample]:
    samples = []
    for filepath in sorted(raw_dir.glob("*.txt")):
        parts = filepath.stem.split('_')
        if len(parts) < 6:
            continue
        pair_id = f"{parts[0]}_{parts[1]}"
        condition = parts[2]
        rep_str = parts[3]
        if rep_str.startswith("rep"):
            rep = int(rep_str[3:])
        else:
            continue
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        text_match = re.search(r'={60}\n\n(.*?)\n\n={60}', content, re.DOTALL)
        if text_match:
            text = text_match.group(1).strip()
        else:
            text = content.strip()
        samples.append(Sample(
            scene_id=filepath.stem,
            condition=condition,
            repeat=rep,
            text=text,
            pair_id=pair_id
        ))
    return samples


def get_pair_a_text(pair_id: str) -> Optional[str]:
    """直接从硬编码字典获取场景 A 文本"""
    return SCENE_A_TEXTS.get(pair_id)


def parse_scores_from_text(text: str) -> Tuple[bool, List[int], str]:
    """
    从 LLM 输出中提取 6 个分数
    返回: (成功, [scores], reason)
    """
    # 方法1: 提取 JSON
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            scores = []
            for dim in DIMENSIONS:
                key = dim + '_score'
                val = data.get(key)
                if val is None:
                    break
                scores.append(int(val))
            if len(scores) == 6:
                return True, scores, ""
        except:
            pass

    # 方法2: 按顺序提取 1-5 数字
    numbers = re.findall(r'\b([1-5])\b', text)
    if len(numbers) >= 6:
        scores = [int(n) for n in numbers[:6]]
        return True, scores, "从文本中提取数字"

    # 方法3: 中文 "空间: 5"
    pattern = r'(?:空间|身体|意图|信息|时间|叙事依赖)\s*[:：]\s*([1-5])'
    matches = re.findall(pattern, text)
    if len(matches) >= 6:
        scores = [int(m) for m in matches[:6]]
        return True, scores, "从中文字段提取"

    return False, [], "无法解析"


async def judge_sample(sample: Sample, scene_a_text: str, debug: bool = False) -> JudgeResult:
    judge_prompt = f"""你是叙事连续性评估专家。

请阅读以下两段场景，判断【场景 B】是否自然承接【场景 A】。

【场景 A】
{scene_a_text[:500]}

【场景 B】
{sample.text[:500]}

请从以下 6 个维度分别评分（1-5），严格按 JSON 格式输出：

{{
    "spatial_score": 1-5,
    "physical_score": 1-5,
    "intentional_score": 1-5,
    "informational_score": 1-5,
    "temporal_score": 1-5,
    "narrative_dependency_score": 1-5,
    "brief_reason": "简短理由"
}}

只输出 JSON，不要有任何额外内容。"""

    async with httpx.AsyncClient(
        trust_env=False,
        timeout=httpx.Timeout(600.0, connect=30.0)
    ) as client:
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": judge_prompt}],
            "temperature": 0.1,
            "max_tokens": 512,
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
                print(f"  [DEBUG] 返回内容长度: {len(content)}")
                print(f"  [DEBUG] 内容预览: {content[:150]}...")
            
            success, scores, reason = parse_scores_from_text(content)
            if success and len(scores) == 6:
                return JudgeResult(
                    scene_id=sample.scene_id,
                    condition=sample.condition,
                    repeat=sample.repeat,
                    spatial_score=scores[0],
                    physical_score=scores[1],
                    intentional_score=scores[2],
                    informational_score=scores[3],
                    temporal_score=scores[4],
                    narrative_dependency_score=scores[5],
                    brief_reason=reason
                )
            
            # 最终尝试：从文本中提取 brief_reason
            reason_match = re.search(r'"brief_reason"\s*[:：]\s*"([^"]*)"', content)
            brief = reason_match.group(1) if reason_match else "解析失败"
            
            return JudgeResult(
                scene_id=sample.scene_id,
                condition=sample.condition,
                repeat=sample.repeat,
                spatial_score=3,
                physical_score=3,
                intentional_score=3,
                informational_score=3,
                temporal_score=3,
                narrative_dependency_score=3,
                brief_reason=brief
            )
            
        except Exception as e:
            print(f"  [ERROR] API 调用失败: {e}")
            return JudgeResult(
                scene_id=sample.scene_id,
                condition=sample.condition,
                repeat=sample.repeat,
                spatial_score=3,
                physical_score=3,
                intentional_score=3,
                informational_score=3,
                temporal_score=3,
                narrative_dependency_score=3,
                brief_reason="API 错误"
            )


async def main():
    base_dir = Path(__file__).parent
    raw_dir = base_dir / "reports" / "raw"
    output_dir = base_dir / "reports" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Phase 5 正式实验评估器")
    print("="*60)
    
    samples = load_samples(raw_dir)
    print(f"找到 {len(samples)} 个样本")
    
    pairs = {}
    for s in samples:
        if s.pair_id not in pairs:
            pairs[s.pair_id] = []
        pairs[s.pair_id].append(s)
    
    print(f"找到 {len(pairs)} 个 pair")
    
    results = []
    for pair_id, pair_samples in pairs.items():
        print(f"\n评估 {pair_id}")
        scene_a_text = get_pair_a_text(pair_id)
        if not scene_a_text:
            print(f"  警告: 找不到 {pair_id} 的场景 A 文本")
            continue
        
        for sample in pair_samples:
            print(f"  评估 {sample.scene_id} (rep={sample.repeat})...")
            result = await judge_sample(sample, scene_a_text, debug=False)
            results.append(result)
            print(f"    S={result.spatial_score} P={result.physical_score} I={result.intentional_score} Info={result.informational_score} T={result.temporal_score} ND={result.narrative_dependency_score}")
    
    # 保存结果
    results_data = [{
        "scene_id": r.scene_id,
        "condition": r.condition,
        "repeat": r.repeat,
        "spatial_score": r.spatial_score,
        "physical_score": r.physical_score,
        "intentional_score": r.intentional_score,
        "informational_score": r.informational_score,
        "temporal_score": r.temporal_score,
        "narrative_dependency_score": r.narrative_dependency_score,
        "brief_reason": r.brief_reason
    } for r in results]
    
    output_path = output_dir / "llm_scores.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    # 统计
    valid = [r for r in results if "解析失败" not in r.brief_reason and "API 错误" not in r.brief_reason]
    print(f"\n有效评分: {len(valid)}/{len(results)} ({len(valid)/len(results)*100:.1f}%)")
    
    if valid:
        print("\n=== 各条件平均分 (Overall) ===")
        conds = ["baseline", "C1", "C2", "C3", "C4"]
        for cond in conds:
            cond_results = [r for r in valid if r.condition == cond]
            if cond_results:
                total = sum(r.spatial_score + r.physical_score + r.intentional_score + r.informational_score + r.temporal_score + r.narrative_dependency_score for r in cond_results)
                avg = total / (len(cond_results) * 6)
                print(f"  {cond}: {avg:.2f} (n={len(cond_results)})")
    else:
        print("没有有效评分，请检查评估过程。")
    
    print(f"\n结果已保存到: {output_path}")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())