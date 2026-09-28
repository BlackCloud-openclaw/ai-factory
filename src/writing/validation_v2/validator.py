"""
ValidationV2 - 四层验证器

关键设计：
- 允许注入 judge/retriever，便于测试
- realm_change 走专用流程（结构字段由程序裁决）
- 其他类型走通用流程（Exact → Alias → Semantic）

C3.4.1 观测修复：
- L4 保留 judgement（供审计追溯）
- 所有分支携带 structural_check（realm 专用）
- 所有 L4 分支标记 fallback_applied=True
- 不改变任何 verdict / matched_layer / confidence 判定

C3.4.2 观测修复：
- 所有分支携带 retrieved_evidence_count / retrieved_evidence_ids
- 三态语义：
    None  → Retriever 未调用（EXACT / ALIAS / realm EXACT_MATCH 短路）
    []    → Retriever 已调用，未命中
    [N]   → Retriever 已调用，命中 N 条
- 不改变任何 verdict / matched_layer / confidence 判定
"""

import logging
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

from .models import (
    VerificationClaim,
    Evidence,
    SemanticJudgement,
    ValidationResultV2,
    Verdict,
    MatchLayer,
)
from .claim_builder import ClaimBuilder
from .evidence_retriever import EvidenceRetriever
from .semantic_judge import SemanticJudge

logger = logging.getLogger(__name__)


TYPE_ALIASES = {
    "plot_flag": ["plot_flag", "flag", "trigger", "标记", "触发", "flag_set", "plot"],
    "knowledge_gain": ["knowledge_gain", "gain_knowledge", "领悟", "理解", "得知", "感悟", "明悟"],
    "inventory_acquire": ["inventory_acquire", "acquire", "获得", "得到", "拿到", "拾取", "收取"],
    "realm_change": ["realm_change", "realm_upgrade", "突破", "晋升", "晋级", "进阶"],
    "location_change": ["location_change", "enter", "进入", "抵达", "来到", "到达"],
    "relationship_change": ["relationship_change", "relationship", "关系", "交恶", "结盟", "和解", "改善", "恶化"],
}


class RealmStructureCheck(str, Enum):
    """realm_change 的结构验证结果"""
    EXACT_MATCH = "exact_match"
    MISMATCH = "mismatch"
    NO_EVENT = "no_event"


class ValidationV2:
    """四层验证器"""

    def __init__(self, judge=None, retriever=None):
        """
        允许注入 judge 和 retriever，便于测试。
        """
        self.retriever = (
            retriever if retriever is not None else EvidenceRetriever()
        )
        self.judge = (
            judge if judge is not None else SemanticJudge()
        )

    # ============================================================
    # 通用匹配辅助
    # ============================================================

    def _match_exact(
        self,
        claim: VerificationClaim,
        event: Dict[str, Any],
    ) -> bool:
        event_type = event.get("type", "")
        if claim.state_change_type != event_type:
            return False
        return self._identity_exact_match(claim, event)

    def _match_alias(
        self,
        claim: VerificationClaim,
        event: Dict[str, Any],
    ) -> bool:
        event_type = event.get("type", "")
        if claim.state_change_type not in TYPE_ALIASES:
            return False
        if event_type not in TYPE_ALIASES[claim.state_change_type]:
            return False
        return self._identity_exact_match(claim, event)

    def _identity_exact_match(
        self,
        claim: VerificationClaim,
        event: Dict[str, Any],
    ) -> bool:
        """
        宽松语义：
        - actor/target 只在两边都有值时要求一致
        - 一边为 None 时跳过该字段的校验
        """
        # actor 校验
        if claim.actor:
            event_actor = event.get("actor")
            if event_actor is not None and claim.actor != event_actor:
                return False

        # target 校验
        if claim.target:
            for key, value in claim.target.items():
                if isinstance(value, str) and value:
                    event_value = event.get(key)
                    if event_value is not None and event_value != value:
                        return False

        return True

    # ============================================================
    # realm_change 专用结构验证
    # ============================================================

    def _check_realm_structure(
        self,
        claim: VerificationClaim,
        writer_events: List[Dict[str, Any]],
    ) -> RealmStructureCheck:
        """
        检查 writer_events 中是否存在精确匹配的 realm 事件。
        """
        target = claim.target or {}
        target_realm = target.get("realm")
        target_stage = target.get("stage")
        actor = claim.actor

        seen_realm_event = False

        for evt in writer_events or []:
            evt_type = evt.get("type", "")
            if evt_type not in ("realm_upgrade", "realm_change"):
                continue

            seen_realm_event = True

            evt_actor = evt.get("actor")
            evt_realm = evt.get("to_major_realm")
            evt_stage = evt.get("to_minor_stage")

            # actor 精确匹配（两边都有值时）
            if actor and evt_actor and actor != evt_actor:
                continue

            # realm 精确匹配（两边都有值时）
            if target_realm and evt_realm and target_realm != evt_realm:
                continue

            # stage 精确匹配（两边都有值时）
            if target_stage and evt_stage and target_stage != evt_stage:
                continue

            # 全部匹配
            actor_ok = (not actor) or (not evt_actor) or (actor == evt_actor)
            realm_ok = (
                (not target_realm)
                or (not evt_realm)
                or (target_realm == evt_realm)
            )
            stage_ok = (
                (not target_stage)
                or (not evt_stage)
                or (target_stage == evt_stage)
            )

            if actor_ok and realm_ok and stage_ok:
                return RealmStructureCheck.EXACT_MATCH

        if seen_realm_event:
            return RealmStructureCheck.MISMATCH

        return RealmStructureCheck.NO_EVENT

    async def _validate_realm_change(
        self,
        claim: VerificationClaim,
        writer_events: List[Dict[str, Any]],
        scene_text: str,
    ) -> ValidationResultV2:
        """
        realm_change 专用验证流程。

        关键不变量：
        - 只有结构精确匹配时才能 SUPPORTED
        - LLM 只能在此流程中判 CONTRADICTED
        - LLM 判 SUPPORTED 不能改变最终判定（仍为 INSUFFICIENT）
        """
        # ---------- 1. 结构验证 ----------
        check = self._check_realm_structure(claim, writer_events)

        if check == RealmStructureCheck.EXACT_MATCH:
            logger.info(
                f"[RealmValidation] EXACT_MATCH for {claim.claim_id}: "
                f"actor={claim.actor}, target={claim.target}"
            )
            return ValidationResultV2(
                claim_id=claim.claim_id,
                state_change_type=claim.state_change_type,
                verdict=Verdict.SUPPORTED,
                matched_layer=MatchLayer.EXACT,
                confidence=1.0,
                reason="Realm structural exact match (actor/realm/stage)",
                evidence_candidates_found=True,
                # C3.4.1 观测
                structural_check=check.value,
                fallback_applied=False,
                # C3.4.2 观测：EXACT 短路，Retriever 未调用
                retrieved_evidence_count=0,
                retrieved_evidence_ids=None,
            )

        # ---------- 2. 语义判断（受结构约束） ----------
        evidences, has_candidates = self.retriever.retrieve(
            claim=claim,
            writer_events=writer_events,
            scene_text=scene_text,
        )
        retrieved_ids = [e.evidence_id for e in evidences]
        retrieved_count = len(evidences)

        judgement: Optional[SemanticJudgement] = None
        if evidences:
            judgement = await self.judge.judge(claim, evidences)

            if judgement.verdict == Verdict.CONTRADICTED:
                logger.info(
                    f"[RealmValidation] CONTRADICTED for {claim.claim_id}: "
                    f"{judgement.reason[:50]}"
                )
                return ValidationResultV2(
                    claim_id=claim.claim_id,
                    state_change_type=claim.state_change_type,
                    verdict=Verdict.CONTRADICTED,
                    matched_layer=MatchLayer.SEMANTIC,
                    confidence=judgement.confidence,
                    judgement=judgement,
                    reason=judgement.reason,
                    evidence_candidates_found=has_candidates,
                    # C3.4.1 观测
                    structural_check=check.value,
                    fallback_applied=False,
                    # C3.4.2 观测：Retriever 已调用
                    retrieved_evidence_count=retrieved_count,
                    retrieved_evidence_ids=retrieved_ids,
                )

        # ---------- 3. 无结构匹配时 → INSUFFICIENT ----------
        if check == RealmStructureCheck.MISMATCH:
            reason = "Realm structural mismatch (actor or realm not matching)"
            logger.info(
                f"[RealmValidation] MISMATCH for {claim.claim_id}: {reason}"
            )
        else:  # NO_EVENT
            if judgement and judgement.verdict == Verdict.SUPPORTED:
                reason = (
                    "Semantic supports realm change but no structural exact match "
                    "(target realm/stage cannot be programmatically verified)"
                )
                logger.info(
                    f"[RealmValidation] LLM SUPPORTED but no structural match "
                    f"for {claim.claim_id}, downgraded to INSUFFICIENT"
                )
            else:
                reason = "No structural exact match for realm_change"
                logger.info(
                    f"[RealmValidation] NO_EVENT for {claim.claim_id}"
                )

        return ValidationResultV2(
            claim_id=claim.claim_id,
            state_change_type=claim.state_change_type,
            verdict=Verdict.INSUFFICIENT,
            matched_layer=MatchLayer.NONE,
            confidence=0.0,
            judgement=judgement,
            reason=reason,
            evidence_candidates_found=has_candidates,
            # C3.4.1 观测
            structural_check=check.value,
            fallback_applied=True,
            # C3.4.2 观测：Retriever 已调用（可能命中为空）
            retrieved_evidence_count=retrieved_count,
            retrieved_evidence_ids=retrieved_ids,
        )

    # ============================================================
    # 主入口
    # ============================================================

    async def validate_claim(
        self,
        claim: VerificationClaim,
        writer_events: List[Dict[str, Any]],
        scene_text: str,
    ) -> ValidationResultV2:
        """验证单个 Claim"""

        # ---------- 特殊路径：realm_change ----------
        if claim.state_change_type == "realm_change":
            return await self._validate_realm_change(
                claim, writer_events, scene_text
            )

        # ---------- 通用路径 ----------
        # C3.4.1: judgement 提升到方法级作用域，供 L4 保留（观测）
        judgement: Optional[SemanticJudgement] = None

        # L1: Exact
        for event in writer_events:
            if self._match_exact(claim, event):
                return ValidationResultV2(
                    claim_id=claim.claim_id,
                    state_change_type=claim.state_change_type,
                    verdict=Verdict.SUPPORTED,
                    matched_layer=MatchLayer.EXACT,
                    confidence=1.0,
                    reason="Exact structural match",
                    evidence_candidates_found=True,
                    # C3.4.1 观测
                    structural_check=None,
                    fallback_applied=False,
                    # C3.4.2 观测：EXACT 短路，Retriever 未调用
                    retrieved_evidence_count=0,
                    retrieved_evidence_ids=None,
                )

        # L2: Alias
        for event in writer_events:
            if self._match_alias(claim, event):
                return ValidationResultV2(
                    claim_id=claim.claim_id,
                    state_change_type=claim.state_change_type,
                    verdict=Verdict.SUPPORTED,
                    matched_layer=MatchLayer.ALIAS,
                    confidence=0.95,
                    reason="Controlled alias match with exact identity",
                    evidence_candidates_found=True,
                    # C3.4.1 观测
                    structural_check=None,
                    fallback_applied=False,
                    # C3.4.2 观测：ALIAS 短路，Retriever 未调用
                    retrieved_evidence_count=0,
                    retrieved_evidence_ids=None,
                )

        # L3: Semantic
        evidences, has_candidates = self.retriever.retrieve(
            claim=claim,
            writer_events=writer_events,
            scene_text=scene_text,
        )
        retrieved_ids = [e.evidence_id for e in evidences]
        retrieved_count = len(evidences)

        if evidences:
            judgement = await self.judge.judge(claim, evidences)

            if judgement.verdict == Verdict.SUPPORTED:
                return ValidationResultV2(
                    claim_id=claim.claim_id,
                    state_change_type=claim.state_change_type,
                    verdict=Verdict.SUPPORTED,
                    matched_layer=MatchLayer.SEMANTIC,
                    confidence=judgement.confidence,
                    judgement=judgement,
                    reason=judgement.reason,
                    evidence_candidates_found=has_candidates,
                    # C3.4.1 观测
                    structural_check=None,
                    fallback_applied=False,
                    # C3.4.2 观测：Retriever 已调用
                    retrieved_evidence_count=retrieved_count,
                    retrieved_evidence_ids=retrieved_ids,
                )

            if judgement.verdict == Verdict.CONTRADICTED:
                return ValidationResultV2(
                    claim_id=claim.claim_id,
                    state_change_type=claim.state_change_type,
                    verdict=Verdict.CONTRADICTED,
                    matched_layer=MatchLayer.SEMANTIC,
                    confidence=judgement.confidence,
                    judgement=judgement,
                    reason=judgement.reason,
                    evidence_candidates_found=has_candidates,
                    # C3.4.1 观测
                    structural_check=None,
                    fallback_applied=False,
                    # C3.4.2 观测：Retriever 已调用
                    retrieved_evidence_count=retrieved_count,
                    retrieved_evidence_ids=retrieved_ids,
                )

        # L4: Insufficient
        # C3.4.1: 保留 judgement 供审计；标记 fallback_applied=True
        # 判定语义不变：verdict / matched_layer / confidence 与修复前一致
        return ValidationResultV2(
            claim_id=claim.claim_id,
            state_change_type=claim.state_change_type,
            verdict=Verdict.INSUFFICIENT,
            matched_layer=MatchLayer.NONE,
            confidence=0.0,
            judgement=judgement,
            reason="No sufficient evidence",
            evidence_candidates_found=has_candidates,
            # C3.4.1 观测
            structural_check=None,
            fallback_applied=True,
            # C3.4.2 观测：Retriever 已调用（可能命中为空）
            retrieved_evidence_count=retrieved_count,
            retrieved_evidence_ids=retrieved_ids,
        )

    async def validate_contract(
        self,
        contract: Dict[str, Any],
        writer_events: List[Dict[str, Any]],
        scene_text: str,
    ) -> List[ValidationResultV2]:
        """验证整个 Contract 的所有 state_changes"""
        results = []
        state_changes = contract.get("observables", {}).get("state_changes", [])
        contract_id = contract.get("scene_id", "")

        for sc in state_changes:
            from src.writing.planning_contract import StateChange
            try:
                change = StateChange(**sc)
            except Exception as e:
                logger.warning(f"Failed to parse StateChange: {e}")
                continue

            claim = ClaimBuilder.from_state_change(change, contract_id)
            result = await self.validate_claim(
                claim, writer_events, scene_text
            )
            results.append(result)

        return results