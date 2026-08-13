"""
Phase 13.2.3B 集成测试 - 验证 ValidatorAgent 与 SemanticValidator 的集成链路
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
from src.writing.validation import (
    SemanticValidator,
    NormalizedMatcher,
    NoOpEmbeddingProvider,
    ValidationEvidence,
)
from src.agents.validator import ValidatorAgent


def create_test_contract(scene_id: str, events: list, source=SignalSource.LLM) -> PlanningContract:
    state_changes = [
        StateChange(
            id=f"evt_{i}",
            type="plot_flag",
            source=source,
            name=evt,
            value=True,
        )
        for i, evt in enumerate(events)
    ]
    return PlanningContract(
        version="1.0",
        scene_id=scene_id,
        intent=Intent(goal="测试目标", conflict="测试冲突", expected_outcome="测试结果"),
        execution=Execution(units=[
            ExecutionUnit(id="U1", label="action", description=e) for e in events
        ]),
        observables=Observables(state_changes=state_changes),
        constraints=[],
        metadata=ContractMetadata(chapter=1, scene_index=0),
    )


class TestEvidenceStability:
    def test_normalized_token_order_stable(self):
        matcher = NormalizedMatcher()
        r1 = matcher.match("突破金丹境界", "突破金丹境界")
        r2 = matcher.match("突破金丹境界", "突破金丹境界")
        assert r1.matched is True
        assert r2.matched is True
        assert r1.matched_text == r2.matched_text

    def test_evidence_id_deterministic(self):
        eid1 = ValidationEvidence.generate_id(
            scene_id="scene_001",
            event_id="evt_001",
            matcher="normalized",
            matched_text="突破 金丹 境界",
        )
        eid2 = ValidationEvidence.generate_id(
            scene_id="scene_001",
            event_id="evt_001",
            matcher="normalized",
            matched_text="突破 金丹 境界",
        )
        assert eid1 == eid2


class TestSourceBlockingPolicy:
    def test_inferred_missing_blocks(self):
        contract = PlanningContract(
            version="1.0",
            scene_id="test_inferred_blocking",
            intent=Intent(goal="测试目标", conflict="测试冲突", expected_outcome="测试结果"),
            execution=Execution(units=[
                ExecutionUnit(id="U1", label="action", description="获得玉佩"),
            ]),
            observables=Observables(state_changes=[
                StateChange(
                    id="evt_001",
                    type="plot_flag",
                    source=SignalSource.LLM,
                    name="获得玉佩",
                    value=True,
                ),
                StateChange(
                    id="evt_002",
                    type="plot_flag",
                    source=SignalSource.INFERRED,
                    name="系统推断事件",
                    value=True,
                ),
            ]),
            constraints=[],
            metadata=ContractMetadata(chapter=1, scene_index=0),
        )

        text = "林逸获得玉佩。"
        validator = SemanticValidator(embedding_provider=NoOpEmbeddingProvider())
        result = validator.validate(contract, text)

        assert result.passed is False
        assert len(result.errors) > 0

    def test_unknown_missing_does_not_block(self):
        contract = PlanningContract(
            version="1.0",
            scene_id="test_unknown_blocking",
            intent=Intent(goal="测试目标", conflict="测试冲突", expected_outcome="测试结果"),
            execution=Execution(units=[
                ExecutionUnit(id="U1", label="action", description="获得玉佩"),
            ]),
            observables=Observables(state_changes=[
                StateChange(
                    id="evt_001",
                    type="plot_flag",
                    source=SignalSource.LLM,
                    name="获得玉佩",
                    value=True,
                ),
                StateChange(
                    id="evt_002",
                    type="plot_flag",
                    source=SignalSource.UNKNOWN,
                    name="旧版遗留标记",
                    value=True,
                ),
            ]),
            constraints=[],
            metadata=ContractMetadata(chapter=1, scene_index=0),
        )

        text = "林逸获得玉佩。"
        validator = SemanticValidator(embedding_provider=NoOpEmbeddingProvider())
        result = validator.validate(contract, text)

        assert result.passed is True
        assert any("旧版" in m for m in result.missing)


class TestValidatorAgentIntegration:
    @pytest.mark.asyncio
    async def test_validator_agent_calls_semantic_validator(self):
        agent = ValidatorAgent()
        contract = create_test_contract("test_agent_integration", ["获得玉佩"])

        # ✅ 文本明确包含连续子串 "获得玉佩"，且长度超过 50 个中文字符
        text = '{"scene_text": "林逸在洞府中获得玉佩。这枚玉佩散发着淡淡灵光，似乎与家族秘密有着密切关联。他仔细端详良久，发现上面刻着奇异符文。这些符文仿佛在诉说一个古老的传说，与青云宗的创派历史似乎有着某种神秘联系。他决定将此事上报长老，但心中隐隐觉得不安。", "events": []}'
        constraints = {"must_events": ["获得玉佩"], "current_state": {}, "active_loop": None}

        result = await agent._validate_novel_enhanced(
            text=text,
            constraints=constraints,
            planning_contract=contract,
        )

        assert "control_scores" in result
        assert "semantic_validation" in result["control_scores"]
        assert result["control_scores"]["semantic_validation"]["passed"] is True
        assert result["control_scores"]["semantic_validation"]["match_count"] >= 1

    @pytest.mark.asyncio
    async def test_validator_contract_immutable(self):
        contract = PlanningContract(
            version="1.0",
            scene_id="test_immutable",
            intent=Intent(goal="测试目标", conflict="测试冲突", expected_outcome="测试结果"),
            execution=Execution(units=[
                ExecutionUnit(id="U1", label="action", description="突破金丹")
            ]),
            observables=Observables(state_changes=[
                StateChange(
                    id="evt_001",
                    type="realm",
                    source=SignalSource.LLM,
                    actor="林逸",
                    to_major_realm="金丹",
                    to_minor_stage=1,
                )
            ]),
            constraints=[],
            metadata=ContractMetadata(chapter=1, scene_index=0),
        )

        before = deepcopy(contract)
        text = "林逸成功突破了金丹境界。"
        validator = SemanticValidator(embedding_provider=NoOpEmbeddingProvider())
        validator.validate(contract, text)

        assert contract.model_dump() == before.model_dump()