# tests/phase14/test_contract_normalizer.py
import pytest
from src.writing.contract_normalizer import ContractNormalizer
from src.writing.planning_contract import (
    PlanningContract,
    Intent,
    Execution,
    ExecutionUnit,
    Observables,
    ContractMetadata,
    SignalSource,
    StateChange,
)


def create_test_contract(units: list, scene_id: str = "test_scene") -> PlanningContract:
    return PlanningContract(
        version="1.0",
        scene_id=scene_id,
        intent=Intent(goal="测试目标", conflict="测试冲突", expected_outcome="测试结果"),
        execution=Execution(units=[
            ExecutionUnit(id=f"U{i}", label="action", description=desc)
            for i, desc in enumerate(units, 1)
        ]),
        observables=Observables(state_changes=[]),
        constraints=[],
        metadata=ContractMetadata(chapter=1, scene_index=0),
    )


class TestContractNormalizer:
    def test_infer_knowledge_gain(self):
        contract = create_test_contract(["发现九宫阵图残片"])
        normalizer = ContractNormalizer()
        normalized = normalizer.normalize(contract)
        assert len(normalized.observables.state_changes) >= 1
        sc = normalized.observables.state_changes[0]
        assert sc.source == SignalSource.INFERRED
        assert sc.type == "knowledge_gain"
        assert sc.confidence == 0.95

    def test_infer_inventory_acquire(self):
        contract = create_test_contract(["获得神秘玉佩"])
        normalizer = ContractNormalizer()
        normalized = normalizer.normalize(contract)
        sc = normalized.observables.state_changes[0]
        assert sc.type == "inventory_acquire"
        assert sc.actor is None
        assert sc.item == "神秘玉佩"

    def test_infer_location_change(self):
        contract = create_test_contract(["进入血煞禁地"])
        normalizer = ContractNormalizer()
        normalized = normalizer.normalize(contract)
        sc = normalized.observables.state_changes[0]
        assert sc.type == "location_change"
        assert sc.location == "血煞禁地"

    def test_infer_realm_change_no_default(self):
        contract = create_test_contract(["突破境界"])
        normalizer = ContractNormalizer()
        normalized = normalizer.normalize(contract)
        sc = normalized.observables.state_changes[0]
        assert sc.type == "realm_change"
        assert sc.to_major_realm is None

    def test_infer_relationship_change_no_default(self):
        contract = create_test_contract(["与张长老交恶"])
        normalizer = ContractNormalizer()
        normalized = normalizer.normalize(contract)
        sc = normalized.observables.state_changes[0]
        assert sc.type == "relationship_change"
        assert sc.delta is None
        assert sc.to_char == "张长老"  # 修正后应提取 "张长老"

    def test_infer_plot_flag(self):
        contract = create_test_contract(["触发上古禁制"])
        normalizer = ContractNormalizer()
        normalized = normalizer.normalize(contract)
        sc = normalized.observables.state_changes[0]
        assert sc.type == "plot_flag"

    def test_multiple_events(self):
        contract = create_test_contract([
            "发现九宫阵图",
            "获得灵丹秘方",
            "进入禁地核心"
        ])
        normalizer = ContractNormalizer()
        normalized = normalizer.normalize(contract)
        assert len(normalized.observables.state_changes) >= 2

    def test_one_event_multiple_changes(self):
        contract = create_test_contract(["发现玉佩并进入秘境"])
        normalizer = ContractNormalizer()
        normalized = normalizer.normalize(contract)
        types = {sc.type for sc in normalized.observables.state_changes}
        assert "knowledge_gain" in types
        assert "location_change" in types

    def test_skip_placeholder(self):
        contract = create_test_contract(["推进主线剧情"])
        normalizer = ContractNormalizer()
        normalized = normalizer.normalize(contract)
        assert len(normalized.observables.state_changes) == 0

    def test_idempotence(self):
        contract = create_test_contract(["获得神秘玉佩"])
        normalizer = ContractNormalizer()
        n1 = normalizer.normalize(contract)
        n2 = normalizer.normalize(n1)
        assert len(n1.observables.state_changes) == len(n2.observables.state_changes)

    def test_llm_signals_preserved(self):
        existing = StateChange(
            id="llm_001",
            type="plot_flag",
            source=SignalSource.LLM,
            confidence=1.0,
            name="existing_plot",
            value=True,
        )
        contract = create_test_contract(["获得神秘玉佩"])
        contract.observables.state_changes = [existing]
        normalizer = ContractNormalizer()
        normalized = normalizer.normalize(contract)
        llm_signals = [sc for sc in normalized.observables.state_changes if sc.source == SignalSource.LLM]
        assert len(llm_signals) == 1
        assert llm_signals[0].id == "llm_001"

    def test_confidence_range(self):
        contract = create_test_contract(["发现上古秘宝"])
        normalizer = ContractNormalizer()
        normalized = normalizer.normalize(contract)
        for sc in normalized.observables.state_changes:
            if sc.source == SignalSource.INFERRED:
                assert 0.0 <= sc.confidence <= 1.0

    def test_no_fact_creation(self):
        contract = create_test_contract(["突破境界"])
        normalizer = ContractNormalizer()
        normalized = normalizer.normalize(contract)
        sc = normalized.observables.state_changes[0]
        assert sc.to_major_realm is None
        assert sc.actor is None

    # 删除 test_contract_with_must_events_field，因为 PlanningContract 无此字段