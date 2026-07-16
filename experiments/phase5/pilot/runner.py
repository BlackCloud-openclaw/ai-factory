#!/usr/bin/env python3
# experiments/phase5/pilot/runner.py

import asyncio
import json
import sys
import re
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import httpx

# 本地导入（不再需要 sys.path 添加）
from runtime_local import NarrativeRuntime
from planning_contract_local import (
    PlanningContract, Intent, Execution, ExecutionUnit,
    Observables, StateChange, ContractMetadata,
    SceneSpecification, WorldSpec, EmotionalArc
)

LLM_API_BASE = "http://localhost:8082/v1"
LLM_MODEL = "Qwen3-32B-Q5_K_M"


def load_scenario(path: Path) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_contract(scenario: Dict) -> PlanningContract:
    data = scenario.copy()
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
        mood=world.get('mood', 'neutral'),
        pacing=world.get('pacing', 'medium'),
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


class PilotRunner:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runtime = NarrativeRuntime(
            llm_api_base=LLM_API_BASE,
            llm_model=LLM_MODEL
        )
        self.results = []
    
    async def run_single(self, scenario: Dict, condition: str, repeat: int) -> Dict:
        contract = build_contract(scenario)
        result = await self.runtime.execute(contract, segments_hint=1)
        
        text = result.full_text.strip()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        scene_id = scenario.get('scene_id', 'P001')
        filename = f"{scene_id}_{condition}_rep{repeat:02d}_{timestamp}.txt"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Scene: {scene_id}\n")
            f.write(f"Condition: {condition}\n")
            f.write(f"Repeat: {repeat}\n")
            f.write(f"{'='*60}\n\n")
            f.write(text)
            f.write(f"\n\n{'='*60}\n")
            f.write(f"Events: {json.dumps(result.all_events, ensure_ascii=False, indent=2)}\n")
        
        return {
            "scene_id": scene_id,
            "condition": condition,
            "repeat": repeat,
            "text": text,
            "events": result.all_events,
            "filepath": str(filepath)
        }
    
    async def run_pair(self, pair_dir: Path, pair_id: str):
        scene_a = load_scenario(pair_dir / "scene_a.yaml")
        
        baseline = load_scenario(pair_dir / "scene_b_baseline.yaml")
        baseline['scene_id'] = f"{pair_id}_baseline"
        
        intervention = load_scenario(pair_dir / "scene_b_intervention.yaml")
        intervention['scene_id'] = f"{pair_id}_intervention"
        
        results = []
        
        for i in range(3):
            print(f"  Running {pair_id} baseline rep {i+1}/3")
            r = await self.run_single(baseline, "baseline", i)
            results.append(r)
        
        for i in range(3):
            print(f"  Running {pair_id} intervention rep {i+1}/3")
            r = await self.run_single(intervention, "intervention", i)
            results.append(r)
        
        return results
    
    async def run_all(self):
        base_dir = Path(__file__).parent
        scenes_dir = base_dir / "scenes"
        all_results = []
        
        for pair_dir in sorted(scenes_dir.iterdir()):
            if pair_dir.is_dir() and pair_dir.name.startswith("pair_"):
                print(f"\n{'='*60}")
                print(f"Running {pair_dir.name}")
                print('='*60)
                results = await self.run_pair(pair_dir, pair_dir.name)
                all_results.extend(results)
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_runs": len(all_results),
            "pairs": len(list(scenes_dir.glob("pair_*"))),
        }
        with open(self.output_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return all_results


async def main():
    base_dir = Path(__file__).parent
    output_dir = base_dir / "reports" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Phase 5 Pilot Runner")
    print("="*60)
    print(f"Output: {output_dir}")
    
    runner = PilotRunner(output_dir)
    await runner.run_all()


if __name__ == "__main__":
    asyncio.run(main())