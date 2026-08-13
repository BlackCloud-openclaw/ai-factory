"""
EvaluationContext Factory：从原始字典构建 EvaluationContext
"""

from typing import Dict, Any, List, Optional
import logging

from src.writing.planning_contract import (
    PlanningContract,
    Intent,
    Execution,
    ExecutionUnit,
    Observables,
    ContractMetadata,
    StateChange,
)
from src.writing.events import event_from_dict
from ..model import EvaluationContext
from .snapshot_builder import SnapshotBuilder, MockSnapshotBuilder
from ..judge.context import JudgeContext

logger = logging.getLogger(__name__)


def create_contract_from_dict(data: Dict[str, Any]) -> PlanningContract:
    units = []
    for u in data.get("execution", {}).get("units", []):
        units.append(ExecutionUnit(**u))
    execution = Execution(units=units)

    intent_data = data.get("intent", {})
    intent = Intent(**intent_data)

    observables_data = data.get("observables", {})
    state_changes = []
    for sc in observables_data.get("state_changes", []):
        state_changes.append(StateChange(**sc))
    observables = Observables(state_changes=state_changes)

    metadata_data = data.get("metadata", {})
    metadata = ContractMetadata(**metadata_data)

    return PlanningContract(
        version=data.get("version", "1.0"),
        scene_id=data.get("scene_id", "test"),
        intent=intent,
        execution=execution,
        observables=observables,
        constraints=[],
        metadata=metadata,
    )


def create_events_from_list(event_list: List[Dict[str, Any]]) -> List:
    events = []
    for evt_dict in event_list:
        evt_type = evt_dict.get("type")
        if not evt_type:
            logger.warning("Event missing 'type' field: %s", evt_dict)
            continue
        evt = event_from_dict(evt_type, evt_dict)
        if evt:
            events.append(evt)
        else:
            logger.warning("Unknown event type '%s', skipping", evt_type)
    return events


def create_context_from_sample(
    sample: Dict[str, Any],
    snapshot_builder: Optional[SnapshotBuilder] = None,
) -> EvaluationContext:
    builder = snapshot_builder or MockSnapshotBuilder()

    # 处理 planning_contract
    contract_data = sample.get("planning_contract", {})
    if not contract_data:
        # 构造默认 PlanningContract
        contract_data = {
            "version": "1.0",
            "scene_id": f"manual_{sample.get('id', 'unknown')}",
            "intent": {
                "goal": "测试目标",
                "conflict": "测试冲突",
                "expected_outcome": "测试结果"
            },
            "execution": {"units": []},
            "observables": {"state_changes": []},
            "constraints": [],
            "metadata": {"chapter": 1, "scene_index": 0},
        }
    # 确保 intent 存在
    if "intent" not in contract_data or not contract_data["intent"]:
        contract_data["intent"] = {
            "goal": "测试目标",
            "conflict": "测试冲突",
            "expected_outcome": "测试结果"
        }

    contract = create_contract_from_dict(contract_data)
    events = create_events_from_list(sample.get("events", []))
    before = builder.build(sample.get("snapshot_before", {}))
    after = builder.build(sample.get("snapshot_after", {}))

    return EvaluationContext(
        planning_contract=contract,
        scene_text=sample.get("scene_before", ""),
        events=events,
        snapshot_before=before,
        snapshot_after=after,
        runtime_metrics=sample.get("runtime_metrics"),
        revision_result=sample.get("revision_result"),
        judge_context=None,
        novel_id=sample.get("id", "manual"),
        volume=0,
        chapter=0,
        scene_idx=0,
    )


def create_contexts(
    samples: List[Dict[str, Any]],
    snapshot_builder: Optional[SnapshotBuilder] = None,
) -> List[EvaluationContext]:
    return [
        create_context_from_sample(s, snapshot_builder=snapshot_builder)
        for s in samples
    ]