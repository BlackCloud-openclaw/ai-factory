from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class Verdict(str, Enum):
    """三态验证结果"""
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT = "INSUFFICIENT"


class MatchLayer(str, Enum):
    """匹配层级"""
    EXACT = "exact"
    ALIAS = "alias"
    SEMANTIC = "semantic"
    NONE = "none"


class EvidenceSource(str, Enum):
    """证据来源"""
    WRITER_EVENT = "writer_event"
    SCENE_TEXT = "scene_text"


@dataclass(frozen=True)
class VerificationClaim:
    """验证命题"""
    claim_id: str
    state_change_type: str
    proposition: str
    actor: Optional[str] = None
    target: Optional[Dict[str, Any]] = None
    source_contract_id: str = ""
    source_state_change_id: str = ""
    # 诊断字段
    raw_state_change_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "state_change_type": self.state_change_type,
            "proposition": self.proposition,
            "actor": self.actor,
            "target": self.target,
            "source_contract_id": self.source_contract_id,
            "source_state_change_id": self.source_state_change_id,
            "raw_state_change_type": self.raw_state_change_type,
        }


@dataclass(frozen=True)
class Evidence:
    """证据"""
    evidence_id: str
    source: EvidenceSource
    text: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source.value,
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SemanticJudgement:
    """语义判断结果"""
    verdict: Verdict
    confidence: float
    reason: str
    evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "evidence_ids": self.evidence_ids,
        }


@dataclass(frozen=True)
class ValidationResultV2:
    """验证结果（B2-2 版本）"""
    claim_id: str
    state_change_type: str
    verdict: Verdict
    matched_layer: MatchLayer
    confidence: float = 0.0
    evidence: Optional[Evidence] = None
    judgement: Optional[SemanticJudgement] = None
    reason: str = ""
    evidence_candidates_found: bool = False

    # ========== C3.4.1 观测字段（不改判定语义） ==========
    # structural_check: realm_change 专用，取值 EXACT_MATCH | MISMATCH | NO_EVENT
    #                   其他类型保持 None
    # fallback_applied: 最终结果是否由 L4 策略兜底（而非 EXACT/ALIAS/SEMANTIC 直接产生）
    structural_check: Optional[str] = None
    fallback_applied: bool = False
    # =====================================================

    # ========== C3.4.2 观测字段（Retriever 层） ==========
    # 三态语义：
    #   None  → Retriever 未被调用（EXACT / ALIAS / realm EXACT_MATCH 短路路径）
    #   []    → Retriever 被调用，但未命中任何证据
    #   [N]   → Retriever 被调用，命中 N 条证据（N >= 1）
    retrieved_evidence_count: int = 0
    retrieved_evidence_ids: Optional[List[str]] = None
    # =====================================================

    @property
    def passed(self) -> bool:
        return self.verdict == Verdict.SUPPORTED

    @property
    def blocked(self) -> bool:
        return self.verdict == Verdict.CONTRADICTED

    @property
    def unknown(self) -> bool:
        return self.verdict == Verdict.INSUFFICIENT

    @property
    def evidence_status(self) -> str:
        if not self.evidence_candidates_found:
            return "NO_EVIDENCE"
        if self.verdict == Verdict.SUPPORTED:
            return "FOUND_SUPPORTED"
        if self.verdict == Verdict.CONTRADICTED:
            return "FOUND_CONTRADICTED"
        return "FOUND_INSUFFICIENT"

    @property
    def llm_invoked(self) -> bool:
        """派生字段：LLM 是否被实际调用（不改判定语义，仅供审计）"""
        return self.judgement is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "state_change_type": self.state_change_type,
            "verdict": self.verdict.value,
            "matched_layer": self.matched_layer.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "judgement": self.judgement.to_dict() if self.judgement else None,
            "evidence_candidates_found": self.evidence_candidates_found,
            "evidence_status": self.evidence_status,
            # ========== C3.4.1 观测字段 ==========
            "structural_check": self.structural_check,
            "fallback_applied": self.fallback_applied,
            "llm_invoked": self.llm_invoked,
            # =====================================
            # ========== C3.4.2 观测字段（Retriever 层） ==========
            "retrieved_evidence_count": self.retrieved_evidence_count,
            "retrieved_evidence_ids": self.retrieved_evidence_ids,
            # =====================================================
        }