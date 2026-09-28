"""
Validator V2 - Deterministic Contract Test

不调用真实 LLM。
- 使用 Mock SemanticJudge
- 100% deterministic
- 验证协议、类型、字段

语义召回能力由 C3.3b LLM Evaluation + Replay 衡量。
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.writing.validation_v2.claim_builder import ClaimBuilder
from src.writing.validation_v2.evidence_retriever import EvidenceRetriever
from src.writing.validation_v2.validator import ValidationV2
from src.writing.validation_v2.models import (
    Verdict,
    MatchLayer,
    SemanticJudgement,
)
from src.writing.planning_contract import StateChange


# ============================================================
# TestClaimBuilder
# ============================================================

class TestClaimBuilder:

    def test_realm_change_claim(self):
        sc = StateChange(
            id="test1",
            type="realm_change",
            actor="林逸",
            to_major_realm="大乘",
            to_minor_stage=1,
        )
        claim = ClaimBuilder.from_state_change(sc, contract_id="c1")

        assert claim.state_change_type == "realm_change"
        assert claim.actor == "林逸"
        assert claim.target == {"realm": "大乘", "stage": 1}
        assert "大乘" in claim.proposition

    def test_realm_change_no_stage(self):
        sc = StateChange(
            id="test2",
            type="realm_change",
            actor="林逸",
            to_major_realm="大乘",
        )
        claim = ClaimBuilder.from_state_change(sc, contract_id="c2")
        assert claim.target["realm"] == "大乘"

    def test_plot_flag_claim(self):
        sc = StateChange(
            id="test3",
            type="plot_flag",
            name="上古封印触发",
            value=True,
        )
        claim = ClaimBuilder.from_state_change(sc, contract_id="c3")
        assert claim.state_change_type == "plot_flag"
        assert "上古封印触发" in claim.proposition

    def test_knowledge_gain_claim(self):
        sc = StateChange(
            id="test4",
            type="knowledge_gain",
            name="阵法原理",
            actor="林逸",
        )
        claim = ClaimBuilder.from_state_change(sc, contract_id="c4")
        assert claim.state_change_type == "knowledge_gain"
        assert "阵法原理" in claim.proposition

    def test_claim_id_is_deterministic(self):
        sc = StateChange(
            id="t1",
            type="plot_flag",
            name="test_flag",
        )
        claim1 = ClaimBuilder.from_state_change(sc, contract_id="c1")
        claim2 = ClaimBuilder.from_state_change(sc, contract_id="c1")
        assert claim1.claim_id == claim2.claim_id

    def test_unknown_type_fallback(self):
        sc = StateChange(
            id="t1",
            type="unknown_type",
        )
        claim = ClaimBuilder.from_state_change(sc, contract_id="c1")
        assert claim.state_change_type == "unknown_type"


# ============================================================
# TestEvidenceRetriever（协议测试，不测语义召回）
# ============================================================

class TestEvidenceRetriever:

    def test_retrieve_returns_tuple(self):
        sc = StateChange(id="t1", type="plot_flag", name="封印触发")
        claim = ClaimBuilder.from_state_change(sc)
        retriever = EvidenceRetriever()

        result = retriever.retrieve(
            claim=claim,
            writer_events=[],
            scene_text="测试文本内容。",
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], bool)

    def test_retrieve_with_identity_in_scene_text(self):
        sc = StateChange(
            id="t1",
            type="plot_flag",
            name="封印触发",
        )
        claim = ClaimBuilder.from_state_change(sc)
        retriever = EvidenceRetriever()

        evidences, found = retriever.retrieve(
            claim=claim,
            writer_events=[],
            scene_text="林逸看见石壁上的封印触发了。",
        )
        # 协议验证：返回类型正确
        assert isinstance(evidences, list)

    def test_evidence_ids_are_traceable(self):
        sc = StateChange(id="t1", type="plot_flag", name="封印触发")
        claim = ClaimBuilder.from_state_change(sc)
        retriever = EvidenceRetriever()

        evidences, _ = retriever.retrieve(
            claim=claim,
            writer_events=[],
            scene_text="封印触发了。",
        )
        for e in evidences:
            assert e.evidence_id  # 非空
            assert isinstance(e.evidence_id, str)

    def test_retrieve_empty_inputs(self):
        sc = StateChange(id="t1", type="plot_flag", name="封印触发")
        claim = ClaimBuilder.from_state_change(sc)
        retriever = EvidenceRetriever()

        evidences, found = retriever.retrieve(
            claim=claim,
            writer_events=None,
            scene_text="",
        )
        assert isinstance(evidences, list)
        assert isinstance(found, bool)


# ============================================================
# TestValidatorContract（注入 Mock Judge）
# ============================================================

class TestValidatorContract:

    @pytest.mark.asyncio
    async def test_validate_claim_with_mock_judge_supported(self):
        mock_judge = AsyncMock()
        mock_judge.judge.return_value = SemanticJudgement(
            verdict=Verdict.SUPPORTED,
            confidence=0.95,
            reason="mock supported",
            evidence_ids=["txt_0"],
        )

        validator = ValidationV2(judge=mock_judge)

        sc = StateChange(id="t1", type="plot_flag", name="封印触发")
        claim = ClaimBuilder.from_state_change(sc)

        result = await validator.validate_claim(
            claim=claim,
            writer_events=[],
            scene_text="封印触发了。",
        )

        assert result.verdict == Verdict.SUPPORTED
        assert result.matched_layer == MatchLayer.SEMANTIC
        assert result.state_change_type == "plot_flag"

    @pytest.mark.asyncio
    async def test_validate_claim_contradicted(self):
        mock_judge = AsyncMock()
        mock_judge.judge.return_value = SemanticJudgement(
            verdict=Verdict.CONTRADICTED,
            confidence=0.92,
            reason="mock contradicted",
            evidence_ids=["txt_0"],
        )

        validator = ValidationV2(judge=mock_judge)

        sc = StateChange(
            id="t1",
            type="plot_flag",
            name="封印触发",
        )
        claim = ClaimBuilder.from_state_change(sc)

        result = await validator.validate_claim(
            claim=claim,
            writer_events=[],
            scene_text="封印并未触发。",
        )

        assert result.verdict == Verdict.CONTRADICTED

    @pytest.mark.asyncio
    async def test_validate_claim_insufficient(self):
        mock_judge = AsyncMock()
        mock_judge.judge.return_value = SemanticJudgement(
            verdict=Verdict.INSUFFICIENT,
            confidence=0.3,
            reason="mock insufficient",
            evidence_ids=[],
        )

        validator = ValidationV2(judge=mock_judge)

        sc = StateChange(id="t1", type="plot_flag", name="封印触发")
        claim = ClaimBuilder.from_state_change(sc)

        result = await validator.validate_claim(
            claim=claim,
            writer_events=[],
            scene_text="林逸打量着石壁。",
        )

        assert result.verdict == Verdict.INSUFFICIENT

    @pytest.mark.asyncio
    async def test_validate_exact_match_no_llm_call(self):
        """Exact 匹配时不应调用 LLM"""
        mock_judge = AsyncMock()
        mock_judge.judge.side_effect = AssertionError(
            "LLM should not be called"
        )

        validator = ValidationV2(judge=mock_judge)

        sc = StateChange(
            id="t1",
            type="plot_flag",
            name="封印触发",
        )
        claim = ClaimBuilder.from_state_change(sc)

        writer_events = [{
            "type": "plot_flag",
            "name": "封印触发",
        }]

        result = await validator.validate_claim(
            claim=claim,
            writer_events=writer_events,
            scene_text="",
        )

        assert result.verdict == Verdict.SUPPORTED
        assert result.matched_layer == MatchLayer.EXACT


# ============================================================
# TestRealmChangeSpecialHandling
# ============================================================

class TestRealmChangeSpecialHandling:
    """realm_change 专用流程测试（全部 Mock，无 LLM）"""

    @pytest.mark.asyncio
    async def test_realm_exact_match_no_llm(self):
        """结构精确匹配时不应调用 LLM"""
        mock_judge = AsyncMock()
        mock_judge.judge.side_effect = AssertionError(
            "LLM should not be called"
        )

        validator = ValidationV2(judge=mock_judge)

        sc = StateChange(
            id="t1",
            type="realm_change",
            actor="林逸",
            to_major_realm="大乘",
            to_minor_stage=1,
        )
        claim = ClaimBuilder.from_state_change(sc)

        writer_events = [{
            "type": "realm_upgrade",
            "actor": "林逸",
            "to_major_realm": "大乘",
            "to_minor_stage": 1,
        }]

        result = await validator.validate_claim(
            claim=claim,
            writer_events=writer_events,
            scene_text="",
        )
        assert result.verdict == Verdict.SUPPORTED
        assert result.matched_layer == MatchLayer.EXACT

    @pytest.mark.asyncio
    async def test_realm_stage_mismatch_direct_insufficient(self):
        """stage 不匹配时不应调用 LLM，直接 INSUFFICIENT"""
        mock_judge = AsyncMock()
        mock_judge.judge.side_effect = AssertionError(
            "LLM should not be called"
        )

        validator = ValidationV2(judge=mock_judge)

        sc = StateChange(
            id="t1",
            type="realm_change",
            actor="林逸",
            to_major_realm="大乘",
            to_minor_stage=1,
        )
        claim = ClaimBuilder.from_state_change(sc)

        writer_events = [{
            "type": "realm_upgrade",
            "actor": "林逸",
            "to_major_realm": "大乘",
            "to_minor_stage": 2,
        }]

        result = await validator.validate_claim(
            claim=claim,
            writer_events=writer_events,
            scene_text="",
        )
        assert result.verdict == Verdict.INSUFFICIENT

    @pytest.mark.asyncio
    async def test_realm_realm_mismatch_direct_insufficient(self):
        """realm 不匹配时不应调用 LLM"""
        mock_judge = AsyncMock()
        mock_judge.judge.side_effect = AssertionError(
            "LLM should not be called"
        )

        validator = ValidationV2(judge=mock_judge)

        sc = StateChange(
            id="t1",
            type="realm_change",
            actor="林逸",
            to_major_realm="大乘",
            to_minor_stage=1,
        )
        claim = ClaimBuilder.from_state_change(sc)

        writer_events = [{
            "type": "realm_upgrade",
            "actor": "林逸",
            "to_major_realm": "化神",
            "to_minor_stage": 1,
        }]

        result = await validator.validate_claim(
            claim=claim,
            writer_events=writer_events,
            scene_text="",
        )
        assert result.verdict == Verdict.INSUFFICIENT

    @pytest.mark.asyncio
    async def test_realm_llm_supported_cannot_override_structure(self):
        """即使 LLM 判 SUPPORTED，无结构匹配时仍返回 INSUFFICIENT"""
        mock_judge = AsyncMock()
        mock_judge.judge.return_value = SemanticJudgement(
            verdict=Verdict.SUPPORTED,
            confidence=0.95,
            reason="mock supported",
            evidence_ids=["txt_0"],
        )

        validator = ValidationV2(judge=mock_judge)

        sc = StateChange(
            id="t1",
            type="realm_change",
            actor="林逸",
            to_major_realm="大乘",
            to_minor_stage=1,
        )
        claim = ClaimBuilder.from_state_change(sc)

        result = await validator.validate_claim(
            claim=claim,
            writer_events=[],
            scene_text="他终于踏入大乘境",
        )

        # 关键：即使 LLM 说 SUPPORTED，也必须是 INSUFFICIENT
        assert result.verdict == Verdict.INSUFFICIENT

    @pytest.mark.asyncio
    async def test_realm_llm_contradicted_is_honored(self):
        """LLM 判 CONTRADICTED 时，即使无结构匹配也应返回 CONTRADICTED"""
        mock_judge = AsyncMock()
        mock_judge.judge.return_value = SemanticJudgement(
            verdict=Verdict.CONTRADICTED,
            confidence=0.92,
            reason="mock contradicted",
            evidence_ids=["txt_0"],
        )

        validator = ValidationV2(judge=mock_judge)

        sc = StateChange(
            id="t1",
            type="realm_change",
            actor="林逸",
            to_major_realm="大乘",
            to_minor_stage=1,
        )
        claim = ClaimBuilder.from_state_change(sc)

        result = await validator.validate_claim(
            claim=claim,
            writer_events=[],
            scene_text="突破失败，依然停留在金丹九层。",
        )

        assert result.verdict == Verdict.CONTRADICTED

    @pytest.mark.asyncio
    async def test_realm_actor_mismatch_insufficient(self):
        """actor 不匹配时不应调用 LLM"""
        mock_judge = AsyncMock()
        mock_judge.judge.side_effect = AssertionError(
            "LLM should not be called"
        )

        validator = ValidationV2(judge=mock_judge)

        sc = StateChange(
            id="t1",
            type="realm_change",
            actor="林逸",
            to_major_realm="大乘",
            to_minor_stage=1,
        )
        claim = ClaimBuilder.from_state_change(sc)

        writer_events = [{
            "type": "realm_upgrade",
            "actor": "苏清雪",
            "to_major_realm": "大乘",
            "to_minor_stage": 1,
        }]

        result = await validator.validate_claim(
            claim=claim,
            writer_events=writer_events,
            scene_text="",
        )
        assert result.verdict == Verdict.INSUFFICIENT