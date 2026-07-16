#!/usr/bin/env python3
# experiments/phase4/runner.py
import asyncio
import json
import yaml
import sys
import re
import copy
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

# 将当前目录添加到 Python 路径
CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR))

# 导入本地模块
import planning_contract_local as pcl
from runtime_local import NarrativeRuntime, RuntimeResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phase4_runner")


@dataclass
class RunResult:
    scene_id: str
    condition: str
    repetition: int
    text: str
    events: List[Dict]
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 场景加载与干预生成
# ============================================================================

def load_scenario(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def apply_intervention(scenario: Dict[str, Any], intervention_type: str) -> Dict[str, Any]:
    data = copy.deepcopy(scenario)
    interventions = data.get('interventions', {})
    
    if intervention_type == 'world':
        world = interventions.get('world', {})
        world['location'] = '墓园'
        world['atmosphere'] = '阴冷，死寂'
        world['sensory'] = ['腐臭', '墓碑', '乌鸦叫']
        interventions['world'] = world
        
    elif intervention_type == 'emotion':
        emotion = interventions.get('reader_emotion', {})
        emotion['begin'] = _reverse_emotion(emotion.get('begin', '好奇'))
        emotion['middle'] = _reverse_emotion(emotion.get('middle', '疑惑'))
        emotion['end'] = _reverse_emotion(emotion.get('end', '不安'))
        interventions['reader_emotion'] = emotion
        
    elif intervention_type == 'function':
        current = interventions.get('narrative_function', 'introduce_mystery')
        interventions['narrative_function'] = _reverse_function(current)
        
    elif intervention_type == 'pov':
        interventions['pov'] = '二叔'
    
    data['interventions'] = interventions
    data['condition'] = f"{intervention_type}_counterfactual"
    return data


def _reverse_emotion(emotion: str) -> str:
    mapping = {
        '好奇': '恐惧', '恐惧': '好奇',
        '疑惑': '狂喜', '狂喜': '疑惑',
        '不安': '释然', '释然': '不安',
        '警惕': '放松', '放松': '警惕',
        '愤怒': '平静', '平静': '愤怒',
        '冷静': '激动', '激动': '冷静',
        '疲惫': '兴奋', '兴奋': '疲惫',
        '坚定': '动摇', '动摇': '坚定',
        '震惊': '平淡', '平淡': '震惊',
        '沉重': '轻快', '轻快': '沉重',
        '期望': '绝望', '绝望': '期望',
        '忐忑': '笃定', '笃定': '忐忑',
        '敬畏': '轻蔑', '轻蔑': '敬畏',
        '悲戚': '欢喜', '欢喜': '悲戚',
        '觉悟': '迷茫', '迷茫': '觉悟',
        '烦躁': '宁静', '宁静': '烦躁',
        '忧虑': '无忧', '无忧': '忧虑',
    }
    return mapping.get(emotion, emotion)


def _reverse_function(func: str) -> str:
    mapping = {
        'introduce_mystery': 'reveal_truth',
        'reveal_truth': 'introduce_mystery',
        'escalate': 'release_tension',
        'release_tension': 'escalate',
        'transition': 'escalate',
    }
    return mapping.get(func, 'escalate')


# ============================================================================
# 构建 Planning Contract
# ============================================================================

def build_contract(scenario: Dict[str, Any]) -> pcl.PlanningContract:
    data = copy.deepcopy(scenario)
    interventions = data.get('interventions', {})
    
    world = interventions.get('world', {})
    emotion = interventions.get('reader_emotion', {})
    func = interventions.get('narrative_function', 'introduce_mystery')
    pov = interventions.get('pov', '林逸')
    
    scene_spec = pcl.SceneSpecification(
        world=pcl.WorldSpec(
            location=world.get('location', '药园'),
            time=world.get('time', '清晨'),
            atmosphere=world.get('atmosphere', '潮湿，雾气低垂'),
            sensory=world.get('sensory', ['药香', '晨雾', '露水'])
        ),
        mood='neutral',
        pacing='medium',
        pov=pov,
        emotional_arc=pcl.EmotionalArc(
            begin=emotion.get('begin', '好奇'),
            middle=emotion.get('middle', '疑惑'),
            end=emotion.get('end', '不安')
        ),
        scene_function=func
    )
    
    units_data = data.get('execution', {}).get('units', [])
    units = [pcl.ExecutionUnit(**u) for u in units_data]
    
    obs_data = data.get('observables', {}).get('state_changes', [])
    state_changes = [pcl.StateChange(**sc) for sc in obs_data]
    observables = pcl.Observables(state_changes=state_changes)
    
    contract = pcl.PlanningContract(
        scene_id=data.get('scene_id', 'S001'),
        intent=pcl.Intent(
            goal=data.get('intent', {}).get('goal', ''),
            conflict=data.get('intent', {}).get('conflict', ''),
            expected_outcome=data.get('intent', {}).get('expected_outcome', '')
        ),
        execution=pcl.Execution(units=units),
        observables=observables,
        constraints=[],
        metadata=pcl.ContractMetadata(chapter=1, scene_index=0),
        scene_spec=scene_spec
    )
    return contract


# ============================================================================
# 实验运行器
# ============================================================================

class Phase4Runner:
    def __init__(self, output_dir: Path = None):
        self.runtime = NarrativeRuntime(
            llm_api_base="http://localhost:8082/v1",
            llm_model="Qwen3-32B-Q5_K_M"
        )
        self.output_dir = output_dir or Path('experiments/phase4/reports/raw')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[RunResult] = []
    
    async def run_single(self, scenario: Dict[str, Any], condition: str, repetition: int) -> RunResult:
        scene_id = scenario.get('scene_id', 'unknown')
        try:
            contract = build_contract(scenario)
            runtime_result = await self.runtime.execute(contract, segments_hint=2)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{scene_id}_{condition}_rep{repetition:02d}_{timestamp}.txt"
            filepath = self.output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Scene: {scene_id}\nCondition: {condition}\nRepetition: {repetition}\nTimestamp: {timestamp}\n{'='*60}\n\n")
                f.write(runtime_result.full_text)
                f.write(f"\n\n{'='*60}\nEvents: {json.dumps(runtime_result.all_events, ensure_ascii=False, indent=2)}\n")
                f.write(f"Segments: {runtime_result.total_segments}, Succeeded: {runtime_result.successful_segments}\n")
            
            return RunResult(
                scene_id=scene_id,
                condition=condition,
                repetition=repetition,
                text=runtime_result.full_text,
                events=runtime_result.all_events,
                success=runtime_result.successful_segments > 0,
                metadata={
                    'total_segments': runtime_result.total_segments,
                    'successful_segments': runtime_result.successful_segments,
                    'fallback_used': runtime_result.fallback_used,
                    'execution_time': runtime_result.execution_time,
                }
            )
        except Exception as e:
            return RunResult(
                scene_id=scene_id,
                condition=condition,
                repetition=repetition,
                text='',
                events=[],
                success=False,
                error=str(e)
            )
    
    async def run_experiment_1(self, scenarios_dir: Path, repetitions: int = 3) -> List[RunResult]:
        all_results = []
        scenario_files = sorted(scenarios_dir.glob('*.yaml'))
        logger.info(f"Experiment 1: {len(scenario_files)} scenarios, {repetitions} reps each")
        
        for filepath in scenario_files:
            scenario = load_scenario(filepath)
            scene_id = scenario.get('scene_id', filepath.stem)
            logger.info(f"--- Scene: {scene_id} ---")
            
            for i in range(repetitions):
                logger.info(f"  Baseline rep {i+1}")
                result = await self.run_single(scenario, 'baseline', i)
                all_results.append(result)
            
            world_scenario = apply_intervention(scenario, 'world')
            for i in range(repetitions):
                logger.info(f"  World counterfactual rep {i+1}")
                result = await self.run_single(world_scenario, 'world_counterfactual', i)
                all_results.append(result)
            
            emotion_scenario = apply_intervention(scenario, 'emotion')
            for i in range(repetitions):
                logger.info(f"  Emotion counterfactual rep {i+1}")
                result = await self.run_single(emotion_scenario, 'emotion_counterfactual', i)
                all_results.append(result)
            
            function_scenario = apply_intervention(scenario, 'function')
            for i in range(repetitions):
                logger.info(f"  Function counterfactual rep {i+1}")
                result = await self.run_single(function_scenario, 'function_counterfactual', i)
                all_results.append(result)
            
            pov_scenario = apply_intervention(scenario, 'pov')
            for i in range(repetitions):
                logger.info(f"  POV counterfactual rep {i+1}")
                result = await self.run_single(pov_scenario, 'pov_counterfactual', i)
                all_results.append(result)
        
        return all_results
    
    async def run_experiment_2(self, scenarios_dir: Path, repetitions: int = 3) -> List[RunResult]:
        all_results = []
        scenario_files = sorted(scenarios_dir.glob('*.yaml'))[:5]
        logger.info(f"Experiment 2: {len(scenario_files)} scenarios")
        
        for filepath in scenario_files:
            scenario = load_scenario(filepath)
            scene_id = scenario.get('scene_id', filepath.stem)
            logger.info(f"--- Scene: {scene_id} ---")
            
            base = copy.deepcopy(scenario)
            base['interventions']['world']['location'] = '药园'
            base['interventions']['world']['sensory'] = ['药香', '晨雾', '露水']
            
            emotion_variants = [
                ('curious', {'begin': '好奇', 'middle': '疑惑', 'end': '不安'}),
                ('fearful', {'begin': '恐惧', 'middle': '紧张', 'end': '绝望'}),
                ('hopeful', {'begin': '期待', 'middle': '坚定', 'end': '希望'}),
            ]
            for name, emo in emotion_variants:
                variant = copy.deepcopy(base)
                variant['interventions']['reader_emotion'] = emo
                variant['condition'] = f"isolation_world_fixed_emotion_{name}"
                for i in range(repetitions):
                    result = await self.run_single(variant, variant['condition'], i)
                    all_results.append(result)
            
            base = copy.deepcopy(scenario)
            base['interventions']['reader_emotion'] = {'begin': '好奇', 'middle': '疑惑', 'end': '不安'}
            
            world_variants = [
                ('garden', {'location': '药园', 'sensory': ['药香', '晨雾', '露水']}),
                ('tomb', {'location': '墓园', 'sensory': ['腐臭', '墓碑', '乌鸦叫']}),
                ('temple', {'location': '祠堂', 'sensory': ['香烛', '旧牌', '灰烬']}),
            ]
            for name, w in world_variants:
                variant = copy.deepcopy(base)
                variant['interventions']['world'] = w
                variant['condition'] = f"isolation_emotion_fixed_world_{name}"
                for i in range(repetitions):
                    result = await self.run_single(variant, variant['condition'], i)
                    all_results.append(result)
        
        return all_results
    
    async def run_experiment_3(self, scenarios_dir: Path, repetitions: int = 3) -> List[RunResult]:
        all_results = []
        scenario_files = sorted(scenarios_dir.glob('*.yaml'))[:3]
        logger.info(f"Experiment 3: {len(scenario_files)} scenarios")
        
        for filepath in scenario_files:
            scenario = load_scenario(filepath)
            scenario['interventions']['world']['location'] = '墓园'
            scenario['interventions']['world']['sensory'] = ['腐臭', '墓碑', '乌鸦叫']
            scenario['interventions']['reader_emotion'] = {'begin': '幸福', 'middle': '喜悦', 'end': '满足'}
            scenario['condition'] = 'conflict_world_vs_emotion'
            for i in range(repetitions):
                result = await self.run_single(scenario, scenario['condition'], i)
                all_results.append(result)
        
        for filepath in scenario_files:
            scenario = load_scenario(filepath)
            scenario['interventions']['narrative_function'] = 'reveal_truth'
            scenario['interventions']['reader_emotion'] = {'begin': '好奇', 'middle': '好奇', 'end': '好奇'}
            scenario['condition'] = 'conflict_function_vs_emotion'
            for i in range(repetitions):
                result = await self.run_single(scenario, scenario['condition'], i)
                all_results.append(result)
        
        for filepath in scenario_files:
            scenario = load_scenario(filepath)
            scenario['interventions']['pov'] = '林逸'
            scenario['interventions']['world']['location'] = '禁地密室'
            scenario['interventions']['world']['sensory'] = ['封印', '铁链', '血迹']
            scenario['condition'] = 'conflict_pov_vs_world'
            for i in range(repetitions):
                result = await self.run_single(scenario, scenario['condition'], i)
                all_results.append(result)
        
        return all_results


# ============================================================================
# 主入口
# ============================================================================

async def main():
    base_dir = Path(__file__).parent
    scenarios_dir = base_dir / 'scenarios' / 'baseline'
    output_dir = base_dir / 'reports' / 'raw'
    
    logger.info("Phase 4 Experiment Runner")
    logger.info(f"Scenarios: {scenarios_dir}")
    logger.info(f"Output: {output_dir}")
    
    if not scenarios_dir.exists():
        logger.error(f"Scenarios directory not found: {scenarios_dir}")
        return
    
    runner = Phase4Runner(output_dir)
    
    logger.info("\n=== Experiment 1: Intervention Effect Test ===")
    results_1 = await runner.run_experiment_1(scenarios_dir, repetitions=3)
    logger.info(f"Experiment 1 complete: {len(results_1)} runs")
    
    logger.info("\n=== Experiment 2: Isolation Test ===")
    results_2 = await runner.run_experiment_2(scenarios_dir, repetitions=3)
    logger.info(f"Experiment 2 complete: {len(results_2)} runs")
    
    logger.info("\n=== Experiment 3: Conflict Test ===")
    results_3 = await runner.run_experiment_3(scenarios_dir, repetitions=3)
    logger.info(f"Experiment 3 complete: {len(results_3)} runs")
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'experiment_1_count': len(results_1),
        'experiment_2_count': len(results_2),
        'experiment_3_count': len(results_3),
        'total_runs': len(results_1) + len(results_2) + len(results_3),
        'failed_runs': sum(1 for r in results_1 + results_2 + results_3 if not r.success)
    }
    summary_path = base_dir / 'reports' / 'summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nSummary: {summary['total_runs']} runs, {summary['failed_runs']} failed")
    logger.info(f"Summary saved: {summary_path}")


if __name__ == '__main__':
    asyncio.run(main())