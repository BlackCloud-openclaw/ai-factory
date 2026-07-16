#!/usr/bin/env python3
# experiments/phase4/quick_test.py
# 修复版 - 使用绝对路径正确加载场景文件

import asyncio
import sys
from pathlib import Path

# 将当前目录添加到 sys.path，确保能导入 runtime_local 和 planning_contract_local
CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR))

import yaml
from planning_contract_local import (
    PlanningContract, Intent, Execution, ExecutionUnit,
    Observables, StateChange, ContractMetadata,
    SceneSpecification, WorldSpec, EmotionalArc
)
from runtime_local import NarrativeRuntime

# 场景文件目录
SCENARIO_DIR = CURRENT_DIR / "scenarios" / "baseline"


def load_scenario(filename: str):
    """加载场景 YAML 文件"""
    path = SCENARIO_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"场景文件不存在: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_contract(scenario_data: dict) -> PlanningContract:
    """从场景数据构建 PlanningContract"""
    data = scenario_data.copy()
    interventions = data.get('interventions', {})
    
    world = interventions.get('world', {})
    emotion = interventions.get('reader_emotion', {})
    func = interventions.get('narrative_function', 'introduce_mystery')
    pov = interventions.get('pov', '林逸')
    
    scene_spec = SceneSpecification(
        world=WorldSpec(
            location=world.get('location', '药园'),
            time=world.get('time', '清晨'),
            atmosphere=world.get('atmosphere', '潮湿，雾气低垂'),
            sensory=world.get('sensory', ['药香', '晨雾', '露水'])
        ),
        mood='neutral',
        pacing='medium',
        pov=pov,
        emotional_arc=EmotionalArc(
            begin=emotion.get('begin', '好奇'),
            middle=emotion.get('middle', '疑惑'),
            end=emotion.get('end', '不安')
        ),
        scene_function=func
    )
    
    units_data = data.get('execution', {}).get('units', [])
    units = [ExecutionUnit(**u) for u in units_data]
    
    obs_data = data.get('observables', {}).get('state_changes', [])
    state_changes = [StateChange(**sc) for sc in obs_data]
    observables = Observables(state_changes=state_changes)
    
    contract = PlanningContract(
        scene_id=data.get('scene_id', 'S001'),
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


async def test_scene(scene_id: str, runtime: NarrativeRuntime, repetitions: int = 3):
    """测试单个场景"""
    print(f"\n{'='*60}")
    print(f"测试 {scene_id}")
    print('='*60)
    
    try:
        scenario = load_scenario(f"{scene_id}.yaml")
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        return
    
    contract = build_contract(scenario)
    
    for i in range(repetitions):
        try:
            result = await runtime.execute(contract, segments_hint=2)
            print(f"  第{i+1}次: 成功={result.successful_segments>0}, 字数={len(result.full_text)}, 段数={result.total_segments}, 成功段={result.successful_segments}, 降级={result.fallback_used}")
        except Exception as e:
            print(f"  第{i+1}次: ❌ 异常: {e}")


async def main():
    """主测试函数"""
    runtime = NarrativeRuntime()
    
    print("\n" + "="*60)
    print("Phase 4 快速测试（稳定性优化验证）")
    print("="*60)
    print(f"场景目录: {SCENARIO_DIR}")
    print(f"LLM API: {runtime.llm_api_base}")
    print(f"LLM 模型: {runtime.llm_model}")
    
    # 测试 S001 (简单场景)
    await test_scene("S001", runtime, repetitions=3)
    
    # 测试 S010 (复杂场景 - 之前完全失败)
    await test_scene("S010", runtime, repetitions=3)
    
    # 测试 S012 (藏书阁场景 - 之前部分失败)
    await test_scene("S012", runtime, repetitions=3)
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


if __name__ == '__main__':
    asyncio.run(main())