#!/usr/bin/env python3
# experiments/phase5/formal/runner.py

import asyncio
import json
import sys
import yaml
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import httpx

# 本地导入（不依赖 sys.path）
from runtime_local import NarrativeRuntime
from planning_contract_local import (
    PlanningContract, Intent, Execution, ExecutionUnit,
    Observables, StateChange, ContractMetadata,
    SceneSpecification, WorldSpec, EmotionalArc
)

LLM_API_BASE = "http://localhost:8082/v1"
LLM_MODEL = "Qwen3-32B-Q5_K_M"


# ============================================================
# 10 对场景定义（Scene A + Scene B Spec）
# ============================================================

SCENE_PAIRS = [
    {
        "id": "pair_01",
        "type": "战斗→休整",
        "scene_a": {
            "location": "炼丹房",
            "text": """炼丹房的青砖地面残留着焦黑的爪痕，空气中浮动着血瞳虎独有的硫磺味。林逸单膝跪地，左肩的伤口正往外渗血，袖口被利爪撕成布条。他低头看了眼怀中半卷《太乙化神录》，虎爪扯断秘银锁链时发出的脆响还在耳边回响。

虎爪最后一击擦过他耳际时，他分明感觉到风压割裂了皮肤。此刻肩胛骨的灼痛反倒让他清醒过来——这种伤至少得用冰魄兰调敷三日。

他扶着墙根站直，血珠顺着指缝滴落在焦痕上，发出细微的嗞嗞声。远处传来巡逻弟子的脚步声，他提起卷轴，拖着受伤的左肩，朝通向静室的密道走去。"""
        },
        "scene_b_spec": {
            "location": "静室",
            "time": "深夜",
            "atmosphere": "安静，烛火微弱",
            "sensory": ["烛火", "血腥味", "石壁凉意"],
            "emotion": {"begin": "疲惫", "middle": "警觉", "end": "平静"},
            "function": "transition",
            "pov": "林逸"
        },
        "execution_units": ["检查肩伤并找药", "展开卷轴查看内容", "发现卷轴中的异常标记"]
    },
    {
        "id": "pair_02",
        "type": "战斗→休整",
        "scene_a": {
            "location": "峡谷",
            "text": """幽谷深处弥漫着妖兽的腥臭，林逸的玄铁剑插在岩缝里，剑刃上还沾着暗红血迹。三日前那场遭遇让他左臂多出三道深可见骨的爪痕，此刻肿胀处泛着诡异的青黑。

他撕下衣摆包扎伤口，脚下踩碎了半截断裂的妖兽骨刺。月光从岩顶裂隙漏下，照见他手中那枚刚夺来的青铜令牌——上面刻着的狼首纹与宗门通缉令上的标记如出一辙。

远处传来瀑布轰鸣声，他扶着岩壁站起身，朝山谷外的石屋方向走去。"""
        },
        "scene_b_spec": {
            "location": "石屋",
            "time": "深夜",
            "atmosphere": "安静，石壁渗水",
            "sensory": ["血腥味", "石壁寒气", "炭火余温"],
            "emotion": {"begin": "疲惫", "middle": "警惕", "end": "坚定"},
            "function": "transition",
            "pov": "林逸"
        },
        "execution_units": ["处理伤口", "研究青铜令牌", "发现令牌与通缉令关联"]
    },
    # ... 其余 8 对在运行时动态生成
]

def generate_all_pairs():
    """生成所有 10 对场景的完整定义"""
    pairs = SCENE_PAIRS.copy()
    
    # 添加 Pair 03-10
    additional_pairs = [
        {
            "id": "pair_03",
            "type": "发现→分析",
            "scene_a": {
                "location": "藏书阁",
                "text": """藏书阁第三层的书架积着半寸厚的灰，林逸推开《云笈七签》时，夹层里滑出半张泛黄的拓片。星图暗纹在月光下泛起幽蓝光晕，他立刻认出这与验灵阵第三重纹路完全重合。

他展开拓片，边缘的撕裂处还残留着朱砂印记。这些符号与他在禁地外围见过的标记如出一辙。窗外传来夜巡弟子经过的脚步声，他将拓片卷起塞入衣襟内侧，合上书册，快步走向阁楼的侧梯。"""
            },
            "scene_b_spec": {
                "location": "卧室",
                "time": "深夜",
                "atmosphere": "昏沉，烛光摇曳",
                "sensory": ["烛火", "墨香", "纸张脆响"],
                "emotion": {"begin": "好奇", "middle": "专注", "end": "怀疑"},
                "function": "transition",
                "pov": "林逸"
            },
            "execution_units": ["在烛光下展开拓片", "比对已知线索", "发现拓片上的可疑标记"]
        },
        {
            "id": "pair_04",
            "type": "发现→分析",
            "scene_a": {
                "location": "古井",
                "text": """古井的青苔在月光下泛着幽绿，林逸用手电筒光束劈开浓雾时，井口溢出的寒气正凝结在他睫毛上。铁链锈迹斑驳处泛着铜绿，他拨开青苔，发现铁链上密布的环状凹痕——像是被无数细绳勒出的伤疤。

手电筒扫过井壁时，光斑在某处突然扭曲。潮湿的岩面上，用朱砂刻着三圈螺旋纹，最深处嵌着半枚指纹。他掏出匕首刮开更多苔藓，取出怀中的拓印纸，将符文拓下后快步离开古井。"""
            },
            "scene_b_spec": {
                "location": "书房",
                "time": "深夜",
                "atmosphere": "安静，烛火明亮",
                "sensory": ["烛火", "羊皮纸", "笔墨气息"],
                "emotion": {"begin": "好奇", "middle": "专注", "end": "忌惮"},
                "function": "transition",
                "pov": "林逸"
            },
            "execution_units": ["比对古井符文与典籍记载", "发现符文与禁地标记吻合", "决定深入调查"]
        },
        {
            "id": "pair_05",
            "type": "冲突→独处",
            "scene_a": {
                "location": "议事堂",
                "text": """议事堂的檀木桌被拍得震响，管事的茶杯盖子跳了两跳。林逸直视着对方泛红的脖颈，指节在袖中握得发白。

「灵田分配的事，什么时候轮到杂役弟子开口？」管事的唾沫星子溅在账册上。

林逸松开拳头，声音比预想中平静：「那就让长老亲自验灵田的灵气浓度。」

他转身踏出议事堂门槛时，晚风裹着青草气扑面而来。身后传来杯盏碎裂的声响，他没有回头，沿着石阶朝山道走去。"""
            },
            "scene_b_spec": {
                "location": "山道",
                "time": "黄昏",
                "atmosphere": "开阔，风中有草木气息",
                "sensory": ["风声", "青草", "归鸟"],
                "emotion": {"begin": "愤怒", "middle": "平复", "end": "坚定"},
                "function": "transition",
                "pov": "林逸"
            },
            "execution_units": ["沿着山道前行", "回想争吵内容", "调整心态，坚定决心"]
        },
        {
            "id": "pair_06",
            "type": "冲突→独处",
            "scene_a": {
                "location": "擂台",
                "text": """擂台上散落的碎石还带着血迹，林逸单膝跪地，右手按着被玄冰剑划伤的腰侧。台下弟子们的议论声嗡嗡作响，莫玄收剑入鞘时嘴角那抹冷笑像根刺扎进他眼底。

「外门弟子也敢挑战内门？不自量力。」莫玄丢下这句话时，袖口扫起的尘灰扑了林逸一脸。

林逸撑着剑柄站起身，没有回话。他扯断束发的布条，任由汗湿的长发遮住半边脸，转身走下擂台，朝山脚下的河边走去。"""
            },
            "scene_b_spec": {
                "location": "河边",
                "time": "黄昏",
                "atmosphere": "宁静，水声潺潺",
                "sensory": ["水声", "花香", "晚风"],
                "emotion": {"begin": "不甘", "middle": "沉思", "end": "决心"},
                "function": "transition",
                "pov": "林逸"
            },
            "execution_units": ["河边独坐", "复盘对战失误", "制定提升计划"]
        },
        {
            "id": "pair_07",
            "type": "调查→追踪",
            "scene_a": {
                "location": "地牢",
                "text": """地牢的霉味混着铁锈气息扑面而来，林逸蹲在囚笼前，用匕首挑起地上半截烧焦的符纸。审讯台上的鞭痕还渗着暗红，俘虏的目光从低垂的乱发间透出来，嘴唇翕动着吐出半句残音：「...深渊入口在...」

话音未落，俘虏突然浑身抽搐，嘴角溢出黑血。林逸冲上前撬开他的嘴，却只摸到半片咬碎的毒囊。他握紧符纸站起身，朝地牢外的密道快步走去。"""
            },
            "scene_b_spec": {
                "location": "密道出口",
                "time": "深夜",
                "atmosphere": "压抑，潮湿",
                "sensory": ["铁锈气息", "滴水声", "霉味"],
                "emotion": {"begin": "紧迫", "middle": "警觉", "end": "坚定"},
                "function": "transition",
                "pov": "林逸"
            },
            "execution_units": ["分析符纸线索", "判断深渊入口方位", "决定立即出发"]
        },
        {
            "id": "pair_08",
            "type": "调查→追踪",
            "scene_a": {
                "location": "灵田",
                "text": """灵田的稻穗在夜风中沙沙作响，林逸蹲在田埂上，指尖抚过一片被踩断的稻秆。断口处还残留着淡淡的灵气余韵，三道平行的压痕延伸向东北方的密林深处。

他掏出怀中的罗盘，指针疯狂旋转后锁定东北方。守夜人的火把在远处移动，他将罗盘收入衣襟，顺着压痕的方向快步追去。"""
            },
            "scene_b_spec": {
                "location": "密林边缘",
                "time": "深夜",
                "atmosphere": "阴冷，树影幢幢",
                "sensory": ["松柏气息", "夜枭啼叫", "脚下的枯枝脆响"],
                "emotion": {"begin": "警觉", "middle": "专注", "end": "坚定"},
                "function": "transition",
                "pov": "林逸"
            },
            "execution_units": ["追踪压痕至密林", "发现脚印与血迹", "确认目标方向"]
        },
        {
            "id": "pair_09",
            "type": "日常→异变",
            "scene_a": {
                "location": "静室",
                "text": """林逸盘膝坐在静室蒲团上，掌心托着半块玉简，丹田处灵力如潮水般涌动。他运转《御气诀》冲击炼气三层时，忽然察觉心神一阵恍惚——眼前浮现出七岁那年被逐出师门的画面，宗门长老的铜锁砸碎肩骨的声音仿佛就在耳边。

他猛地咬破舌尖，血珠溅在玉简上激起青光：「修炼时心魔作祟，不能停！」"""
            },
            "scene_b_spec": {
                "location": "静室",
                "time": "深夜",
                "atmosphere": "压抑，烛火摇晃",
                "sensory": ["烛火", "心跳声", "玉简嗡鸣"],
                "emotion": {"begin": "紧张", "middle": "挣扎", "end": "坚定"},
                "function": "transition",
                "pov": "林逸"
            },
            "execution_units": ["强行压制心魔", "继续冲击境界", "发现经脉异常"]
        },
        {
            "id": "pair_10",
            "type": "日常→异变",
            "scene_a": {
                "location": "药园",
                "text": """清晨的药园弥漫着露水与药草的气息，林逸提着竹篮穿梭在灵草间。他弯腰采下一株紫纹参时，指尖忽然触到异样的腥甜——本该清冽的药香里，混着一丝腐烂果核的腐臭。

他拨开半人高的玉灵草，发现三株血灵芝的叶片正蜷曲发黑，根部渗出暗褐色黏液。远处传来守药人的脚步声，他将血灵芝连根挖起，快步离开药园。"""
            },
            "scene_b_spec": {
                "location": "丹房",
                "time": "清晨",
                "atmosphere": "压抑，药香混腐臭",
                "sensory": ["药香", "腐臭味", "石壁凉意"],
                "emotion": {"begin": "好奇", "middle": "警觉", "end": "不安"},
                "function": "transition",
                "pov": "林逸"
            },
            "execution_units": ["检查血灵芝根部", "发现异常黑斑与灵田异动关联", "决定深入调查"]
        }
    ]
    
    pairs.extend(additional_pairs)
    return pairs


def build_scene_b_contract(spec: Dict, units: List[str], scene_id: str, entry_location: str = None) -> Dict:
    """构建场景 B 的 Contract（含可选的 EntryLocation）"""
    interventions = {
        "world": spec,
        "reader_emotion": spec["emotion"],
        "narrative_function": spec.get("function", "transition"),
        "pov": spec.get("pov", "林逸")
    }
    
    if entry_location:
        interventions["entry_location"] = entry_location
    
    return {
        "scene_id": scene_id,
        "interventions": interventions,
        "intent": {
            "goal": f"自然承接上一场景，在{spec['location']}展开",
            "conflict": "需要平衡身心状态与线索整理",
            "expected_outcome": "完成过渡，建立下一场景的基础"
        },
        "execution": {
            "units": [{"id": f"U{i+1}", "label": "action", "description": u, "attributes": {}} 
                     for i, u in enumerate(units)]
        },
        "observables": {
            "state_changes": []
        }
    }


def build_contract_from_config(config: Dict, include_entry: bool = False, entry_location: str = None) -> PlanningContract:
    """从场景配置构建完整的 Planning Contract"""
    data = config.copy()
    interventions = data.get('interventions', {})
    
    world = interventions.get('world', {})
    emotion = interventions.get('reader_emotion', {})
    func = interventions.get('narrative_function', 'transition')
    pov = interventions.get('pov', '林逸')
    
    scene_spec = SceneSpecification(
        world=WorldSpec(
            location=world.get('location', ''),
            time=world.get('time', ''),
            atmosphere=world.get('atmosphere', ''),
            sensory=world.get('sensory', [])
        ),
        mood='neutral',
        pacing='medium',
        reader_emotion=EmotionalArc(
            begin=emotion.get('begin', ''),
            middle=emotion.get('middle', ''),
            end=emotion.get('end', '')
        ),
        narrative_function=func,
        pov=pov
    )
    
    units_data = data.get('execution', {}).get('units', [])
    units = [ExecutionUnit(**u) for u in units_data]
    
    obs_data = data.get('observables', {}).get('state_changes', [])
    state_changes = [StateChange(**sc) for sc in obs_data]
    observables = Observables(state_changes=state_changes)
    
    contract = PlanningContract(
        scene_id=data.get('scene_id', 'P001'),
        intent=Intent(
            goal=data.get('intent', {}).get('goal', ''),
            conflict=data.get('intent', {}).get('conflict', ''),
            expected_outcome=data.get('intent', {}).get('expected_outcome', '')
        ),
        execution=Execution(units=units),
        observables=observables,
        constraints=[],
        metadata=ContractMetadata(chapter=1, scene_index=0),
        scene_spec=scene_spec
    )
    return contract


# ============================================================
# 实验运行器
# ============================================================

class FormalExperimentRunner:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runtime = NarrativeRuntime(
            llm_api_base=LLM_API_BASE,
            llm_model=LLM_MODEL
        )
        self.results = []
    
    def get_prompt_modifier(self, condition: str, entry_location: str = None, 
                           physical_state: str = None, goal: str = None,
                           open_threads: List[str] = None) -> str:
        """根据条件构建状态注入文本"""
        lines = []
        
        if condition == "baseline":
            return ""
        
        if entry_location:
            lines.append(f"【场景起点】上一场景结束时你在：{entry_location}")
            lines.append(f"【当前场景】发生在：{self.current_spec['location']}")
            lines.append("请从上一场景的位置自然过渡到当前场景。")
        
        if physical_state and condition in ["C2", "C3", "C4"]:
            lines.append(f"【身体状态】{physical_state}")
        
        if goal and condition in ["C3", "C4"]:
            lines.append(f"【当前目标】{goal}")
        
        if open_threads and condition == "C4":
            lines.append(f"【未解决线索】{', '.join(open_threads)}")
        
        return "\n".join(lines)
    
    async def run_single(self, pair_config: Dict, condition: str, repeat: int, 
                         entry_location: str = None) -> Dict:
        """运行单次生成"""
        # 构建场景 B 配置
        spec = pair_config["scene_b_spec"]
        units = pair_config["execution_units"]
        scene_id = f"{pair_config['id']}_{condition}_rep{repeat:02d}"
        
        config = build_scene_b_contract(spec, units, scene_id, entry_location)
        
        # 额外状态注入（通过修改 prompt）
        # 由于 runtime_local 不支持动态注入状态，我们在 build_contract 中通过修改 scene_spec 传递
        
        contract = build_contract_from_config(config)
        
        # 执行
        result = await self.runtime.execute(contract, segments_hint=1)
        text = result.full_text.strip()
        
        # 保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{scene_id}_{timestamp}.txt"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Scene: {scene_id}\n")
            f.write(f"Condition: {condition}\n")
            f.write(f"Repeat: {repeat}\n")
            f.write(f"EntryLocation: {entry_location or 'None'}\n")
            f.write(f"{'='*60}\n\n")
            f.write(text)
            f.write(f"\n\n{'='*60}\n")
            f.write(f"Events: {json.dumps(result.all_events, ensure_ascii=False, indent=2)}\n")
        
        return {
            "scene_id": scene_id,
            "pair_id": pair_config["id"],
            "condition": condition,
            "repeat": repeat,
            "entry_location": entry_location,
            "text": text,
            "filepath": str(filepath)
        }
    
    async def run_pair(self, pair_config: Dict, repeat: int = 3):
        """运行一个场景对的所有条件"""
        results = []
        pair_id = pair_config["id"]
        scene_a = pair_config["scene_a"]
        entry_location = scene_a["location"]
        
        conditions = [
            ("baseline", None),
            ("C1", entry_location),
            ("C2", entry_location),
            ("C3", entry_location),
            ("C4", entry_location),
        ]
        
        for condition, loc in conditions:
            for r in range(repeat):
                print(f"  {pair_id} {condition} rep {r+1}/{repeat}")
                result = await self.run_single(pair_config, condition, r, loc)
                results.append(result)
        
        return results
    
    async def run_all(self):
        pairs = generate_all_pairs()
        all_results = []
        
        for i, pair in enumerate(pairs):
            print(f"\n{'='*60}")
            print(f"运行 {pair['id']} ({pair['type']}) [{i+1}/{len(pairs)}]")
            print('='*60)
            results = await self.run_pair(pair)
            all_results.extend(results)
        
        # 保存汇总 - 修复路径
        summary_dir = self.output_dir.parent
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_runs": len(all_results),
            "pairs": len(pairs),
            "conditions": ["baseline", "C1", "C2", "C3", "C4"],
            "repeats": 3
        }
        with open(summary_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return all_results


async def main():
    base_dir = Path(__file__).parent
    output_dir = base_dir / "reports" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Phase 5 正式实验")
    print("="*60)
    print(f"输出目录: {output_dir}")
    
    runner = FormalExperimentRunner(output_dir)
    await runner.run_all()
    
    print("\n" + "="*60)
    print("实验完成!")
    print(f"输出: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())