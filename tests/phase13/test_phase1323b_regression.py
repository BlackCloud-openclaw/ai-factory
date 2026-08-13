"""
Phase 13.2.3B 回归测试
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
    SignalWeightPolicy,
    ExactMatcher,
    NormalizedMatcher,
    KeywordCoverageMatcher,
    ValidationEvidence,
    NoOpEmbeddingProvider,
)


def create_test_contract(scene_id: str, events: list) -> PlanningContract:
    state_changes = [
        StateChange(
            id=f"evt_{i}",
            type="plot_flag",
            source=SignalSource.LLM,
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


class TestMatcherStability:
    def test_exact_matcher(self):
        matcher = ExactMatcher()
        result = matcher.match("金丹突破", "林逸在洞府中完成了金丹突破")
        assert result.matched is True
        assert result.confidence == 1.0

    def test_normalized_matcher(self):
        matcher = NormalizedMatcher()
        result = matcher.match("突破金丹", "林逸在洞府中突破了金丹境界")
        assert result.matched is True
        assert result.confidence == 0.9

    def test_keyword_coverage_matcher(self):
        """测试 KeywordCoverageMatcher 在完全匹配时的行为"""
        matcher = KeywordCoverageMatcher()
        result = matcher.match(
            "探查禁地发现遗骸",
            "探查禁地发现遗骸"   # 完全相同
        )
        assert result.matched is True
        assert result.confidence >= 0.9


class TestValidatorDeterministic:
    def test_same_input_same_output(self):
        contract = create_test_contract(
            "test_deterministic",
            ["获得神秘玉佩", "突破金丹境界"],
        )
        text = "林逸在秘境中获得了神秘玉佩，随后突破了金丹境界。"

        validator = SemanticValidator(embedding_provider=NoOpEmbeddingProvider())
        result1 = validator.validate(contract, text)
        result2 = validator.validate(contract, text)

        assert result1.passed == result2.passed
        assert len(result1.matched) == len(result2.matched)
        for e1, e2 in zip(result1.matched, result2.matched):
            assert e1.evidence_id == e2.evidence_id


class TestEvidence:
    def test_evidence_id_generation(self):
        eid = ValidationEvidence.generate_id(
            scene_id="scene_001",
            event_id="evt_001",
            matcher="exact",
            matched_text="金丹突破",
        )
        assert len(eid) == 12
        assert eid.isalnum()


class TestValidationResult:
    def test_validation_version_present(self):
        contract = create_test_contract("test_version", ["获得物品"])
        text = "林逸获得了物品。"
        validator = SemanticValidator(embedding_provider=NoOpEmbeddingProvider())
        result = validator.validate(contract, text)
        assert result.validation_version == "13.2.3B-v1.1"


class TestSignalWeightPolicy:
    def test_default_weights(self):
        policy = SignalWeightPolicy()
        assert policy.weight(SignalSource.LLM) == 1.0
        assert policy.weight(SignalSource.INFERRED) == 0.6
        assert policy.weight(SignalSource.UNKNOWN) == 0.3

    def test_weighted_score(self):
        policy = SignalWeightPolicy()
        score = policy.weighted_score(0.8, SignalSource.INFERRED)
        assert score == 0.48