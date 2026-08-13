# tests/phase13/test_phase1323a_freeze.py
"""
Phase 13.2.3A 冻结回归测试
锁定 ContractNormalizer + EventClassifier + StateChangeFactory 的确定性行为。
"""

import pytest
from copy import deepcopy

from src.writing.planning_contract import (
    PlanningContract,
    Intent,
    Execution,
    ExecutionUnit,
    Observables,
    ContractMetadata,
    StateChange,
    SignalSource,
)
from src.writing.contract_normalizer import ContractNormalizer
from src.writing.event_classifier import EventClassifier, EventType
from src.writing.state_change_factory import StateChangeFactory


# ---- 测试辅助 ----

def create_test_contract(
    scene_id: str,
    units: list,
    existing_changes: list = None,
) -> PlanningContract:
    """创建标准测试用 Contract。"""
    return PlanningContract(
        version="1.0",
        scene_id=scene_id,
        intent=Intent(
            goal="测试目标",
            conflict="测试冲突",
            expected_outcome="测试结果"
        ),
        execution=Execution(units=[
            ExecutionUnit(id=f"U{i}", label="action", description=desc)
            for i, desc in enumerate(units, 1)
        ]),
        observables=Observables(
            state_changes=existing_changes or []
        ),
        constraints=[],
        metadata=ContractMetadata(chapter=1, scene_index=0),
    )


# ---- 测试用例 ----

class TestFreezeProvenance:
    """A-10 / A-2: 来源不可篡改"""

    def test_llm_signal_source_preserved(self):
        """LLM 信号必须保留 source=llm，不受 Normalizer 影响"""
        llm_change = StateChange(
            id="llm_001",
            type="realm",
            source=SignalSource.LLM,
            actor="林逸",
            to_major_realm="金丹",
            to_minor_stage=1,
        )
        contract = create_test_contract(
            scene_id="test_provenance_001",
            units=["突破金丹境界"],
            existing_changes=[llm_change],
        )

        normalizer = ContractNormalizer()
        normalized = normalizer.normalize(contract)

        matched = [sc for sc in normalized.observables.state_changes if sc.id == "llm_001"]
        assert len(matched) == 1
        assert matched[0].source == SignalSource.LLM
        assert matched[0].to_major_realm == "金丹"

        logs = normalizer.get_audit_logs()
        last_log = logs[-1]
        assert last_log["signals"]["state_changes"]["llm"] >= 1

    def test_inferred_signal_source_marked(self):
        """推断信号必须标记为 INFERRED"""
        contract = create_test_contract(
            scene_id="test_provenance_002",
            units=["获得神秘玉佩"],
            existing_changes=[],
        )

        normalizer = ContractNormalizer()
        normalized = normalizer.normalize(contract)

        inferred = [
            sc for sc in normalized.observables.state_changes
            if sc.source == SignalSource.INFERRED
        ]
        assert len(inferred) >= 1
        assert inferred[0].item == "神秘玉佩"

        logs = normalizer.get_audit_logs()
        last_log = logs[-1]
        assert last_log["signals"]["state_changes"]["inferred"] >= 1


class TestFreezeUnknownSource:
    """A-10: UNKNOWN 策略"""

    def test_unknown_source_default(self):
        """未声明来源的 StateChange 默认为 UNKNOWN"""
        sc = StateChange(id="unknown_001", type="realm")
        assert sc.source == SignalSource.UNKNOWN

    def test_unknown_source_not_treated_as_llm(self):
        """UNKNOWN 来源不应被误认为 LLM，且 Normalizer 应正确记录来源。"""
        contract = create_test_contract(
            scene_id="test_unknown_001",
            units=["获得神秘玉佩"],  # 触发推断，使 Normalizer 记录审计
            existing_changes=[
                StateChange(
                    id="manual_001",
                    type="realm",
                    source=SignalSource.UNKNOWN,
                    actor="系统",
                    to_major_realm="筑基",
                    to_minor_stage=1,
                )
            ],
        )

        normalizer = ContractNormalizer()
        normalized = normalizer.normalize(contract)

        # 验证原始 UNKNOWN 信号保留
        unknown_signal = next(
            sc for sc in normalized.observables.state_changes
            if sc.id == "manual_001"
        )
        assert unknown_signal.source == SignalSource.UNKNOWN

        # 验证推断信号已被添加（确保 Normalizer 有工作）
        inferred = [
            sc for sc in normalized.observables.state_changes
            if sc.source == SignalSource.INFERRED
        ]
        assert len(inferred) > 0

        # 审计日志应包含本次操作，且 llm 计数为 0（无 LLM 信号）
        logs = normalizer.get_audit_logs()
        assert len(logs) > 0
        last_log = logs[-1]
        assert last_log["signals"]["state_changes"]["llm"] == 0


class TestFreezeMultiEvent:
    """A-11: 多事件确定性"""

    def test_multi_event_classification(self):
        """一个 must_event 产生多个 EventType"""
        text = "林逸进入秘境获得玄天古玉"
        result = EventClassifier.classify(text)

        assert EventType.LOCATION_CHANGE in result
        assert EventType.ITEM_ACQUIRE in result
        assert len(result) == len(set(result))

    def test_multi_event_propagation(self):
        """多个 EventType 产生多个 StateChange"""
        contract = create_test_contract(
            scene_id="test_multi_001",
            units=["进入秘境获得玄天古玉"],
            existing_changes=[],
        )

        normalizer = ContractNormalizer()
        normalized = normalizer.normalize(contract)

        types = {sc.type for sc in normalized.observables.state_changes}
        assert "location" in types
        assert "inventory" in types

    def test_multi_event_deterministic_order(self):
        """相同输入产生相同输出顺序"""
        text = "进入秘境发现真相获得古玉"
        result1 = EventClassifier.classify(text)
        result2 = EventClassifier.classify(text)

        assert result1 == result2


class TestFreezeHashBoundary:
    """A-12: hash 不依赖输出"""

    def test_hash_boundary_stable(self):
        """Hash 在 Normalizer 前后不变（不依赖 observables）"""
        contract = create_test_contract(
            scene_id="test_hash_001",
            units=["获得神秘玉佩"],
            existing_changes=[],
        )

        normalizer = ContractNormalizer()
        initial_hash = normalizer._compute_input_hash(contract)

        normalized = normalizer.normalize(contract)
        after_hash = normalized.enrichment.input_hash

        assert initial_hash == after_hash
        assert len(normalized.observables.state_changes) > 0

    def test_hash_unchanged_after_repeat(self):
        """重复 Normalize 不改变 hash"""
        contract = create_test_contract(
            scene_id="test_hash_002",
            units=["发现灭门真相"],
            existing_changes=[],
        )

        normalizer = ContractNormalizer()
        normalized1 = normalizer.normalize(contract)
        normalized2 = normalizer.normalize(normalized1)

        assert normalized1.enrichment.input_hash == normalized2.enrichment.input_hash
        assert len(normalized1.observables.state_changes) == len(
            normalized2.observables.state_changes
        )


class TestFreezeIdempotence:
    """A-3/A-12: 幂等性综合验证"""

    def test_normalizer_idempotent(self):
        """normalize(normalize(c)) == normalize(c)"""
        contract = create_test_contract(
            scene_id="test_idempotent_001",
            units=["获得神秘玉佩", "突破金丹境界"],
            existing_changes=[],
        )

        normalizer = ContractNormalizer()
        n1 = normalizer.normalize(contract)
        n2 = normalizer.normalize(n1)

        def sort_by_id(changes):
            return sorted(changes, key=lambda x: x.id)

        sorted1 = sort_by_id(n1.observables.state_changes)
        sorted2 = sort_by_id(n2.observables.state_changes)

        assert len(sorted1) == len(sorted2)
        for sc1, sc2 in zip(sorted1, sorted2):
            assert sc1.id == sc2.id
            assert sc1.type == sc2.type
            assert sc1.source == sc2.source

        assert n1.enrichment.input_hash == n2.enrichment.input_hash

    def test_normalizer_does_not_modify_llm_fields(self):
        """A-9: LLM 字段不变"""
        original_llm = StateChange(
            id="llm_fixed",
            type="realm",
            source=SignalSource.LLM,
            actor="林逸",
            to_major_realm="元婴",
            to_minor_stage=1,
        )
        contract = create_test_contract(
            scene_id="test_invariant_001",
            units=["突破元婴境界"],
            existing_changes=[original_llm],
        )

        before = deepcopy(original_llm)
        normalizer = ContractNormalizer()
        normalized = normalizer.normalize(contract)

        matched = next(sc for sc in normalized.observables.state_changes if sc.id == "llm_fixed")
        assert matched.actor == before.actor
        assert matched.to_major_realm == before.to_major_realm
        assert matched.to_minor_stage == before.to_minor_stage


class TestFreezeValidatorContract:
    """A-8: Validator 可消费性验证"""

    def test_validator_contract_structure(self):
        """确保 Normalizer 输出符合 Validator 消费要求"""
        contract = create_test_contract(
            scene_id="test_validator_001",
            units=["获得玄天古玉", "进入秘境", "突破金丹"],
            existing_changes=[],
        )

        normalizer = ContractNormalizer()
        normalized = normalizer.normalize(contract)

        assert normalized.observables is not None
        assert len(normalized.observables.state_changes) > 0

        for sc in normalized.observables.state_changes:
            assert sc.id is not None and len(sc.id) > 0
            assert sc.source is not None
            assert sc.id in normalized.enrichment.sources
            # 来源应是有效的枚举值
            assert normalized.enrichment.sources[sc.id] in [
                "llm", "inferred", "unknown", "system", "normalized"
            ]

        logs = normalizer.get_audit_logs()
        last_log = logs[-1]
        assert "signals" in last_log
        assert last_log["signals"]["state_changes"]["total"] == len(
            normalized.observables.state_changes
        )