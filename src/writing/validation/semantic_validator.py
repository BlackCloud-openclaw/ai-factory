# src/writing/validation/semantic_validator.py
"""
Semantic Validator - 主管道

Phase 13.2.3B: 整合所有匹配器，输出 ValidationResult
Phase 13.2.3C: 填充 blocking_missing 字段
Patch-1.1: 策略 B (INFERRED 阻断), embedding confidence gate
"""

from typing import List, Optional, Dict, Any

from src.writing.planning_contract import PlanningContract, StateChange, SignalSource
from .evidence import ValidationEvidence, ValidationResult
from .signal_weight import SignalWeightPolicy
from .matchers import (
    Matcher,
    ExactMatcher,
    NormalizedMatcher,
    KeywordCoverageMatcher,
    MatcherResult,
)
from .embedding_provider import EmbeddingProvider
from .embedding_matcher import EmbeddingMatcher, NoOpEmbeddingProvider
import logging
from .models import MissingContractChange, ContractSeverity


logger = logging.getLogger(__name__)

logger.critical("D5.1 SemanticValidator module loaded (with missing_changes support)")

class SemanticValidator:
    def __init__(
        self,
        signal_weight_policy: Optional[SignalWeightPolicy] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        keyword_threshold: float = 0.6,
        embedding_threshold: float = 0.30,
        embedding_min_confidence: float = 0.6,
        enable_embedding: bool = True,
    ):
        logger.critical("D5.1 SemanticValidator instance created")
        self._weight_policy = signal_weight_policy or SignalWeightPolicy()
        self._keyword_threshold = keyword_threshold
        self._embedding_threshold = embedding_threshold
        self._embedding_min_confidence = embedding_min_confidence
        self._enable_embedding = enable_embedding

        self._matchers: List[Matcher] = [
            ExactMatcher(),
            NormalizedMatcher(),
            KeywordCoverageMatcher(threshold=keyword_threshold),
        ]

        provider = embedding_provider or NoOpEmbeddingProvider()
        self._embedding_matcher = EmbeddingMatcher(
            provider=provider,
            threshold=embedding_threshold,
            enable=enable_embedding,
        )

    def validate(
        self,
        contract: PlanningContract,
        scene_text: str,
    ) -> ValidationResult:
        """
        执行验证，返回包含 blocking_missing 的 ValidationResult。
        """
        logger.critical("D5.1 SemanticValidator validate called with missing_changes support")
        missing_changes = []   # 新增
        
        state_changes = contract.observables.state_changes
        logger.info(
            "SemanticValidator start: scene_id=%s, state_changes=%d",
            contract.scene_id,
            len(contract.observables.state_changes)
        )

        if not state_changes:
            return ValidationResult(
                passed=True,
                missing=[],
                matched=[],
                blocking_missing=[],  # Phase 13.2.3C
                overall_confidence=1.0,
                weight_applied=1.0,
                errors=["No state_changes to validate"],
            )

        missing = []
        blocking_missing = []  # Phase 13.2.3C
        matched_evidences = []
        total_weighted = 0.0
        total_count = 0

        for sc in state_changes:
            event_text = self._get_event_text(sc)
            if not event_text:
                continue

            matched, evidence = self._run_pipeline(
                sc, event_text, scene_text, contract.scene_id
            )

            if matched:
                matched_evidences.append(evidence)
                total_weighted += evidence.weight
                total_count += 1
            else:
                missing.append(event_text)
                # Phase 13.2.3C: 策略 B (LLM, SYSTEM, INFERRED 阻断)
                if sc.source in (
                    SignalSource.LLM,
                    SignalSource.SYSTEM,
                    SignalSource.INFERRED,
                ):
                    blocking_missing.append(event_text)

                # 新增：生成 MissingContractChange 投影
                change_type = self._contract_type_value(sc)
                missing_changes.append(
                    MissingContractChange(
                        type=change_type,
                        description=self._describe_state_change(sc),
                        severity=ContractSeverity.BLOCKING,
                        actor=getattr(sc, 'actor', None),
                        fields=sc.model_dump() if hasattr(sc, 'model_dump') else {},
                        source=sc.source.value if hasattr(sc.source, 'value') else str(sc.source),
                        contract_id=getattr(contract, 'scene_id', None),
                        confidence=evidence.confidence if evidence else 0.0,  # 修正
                    )
                )
                # 日志埋点
                logger.info(
                    "CONTRACT_MISSING_CHANGE: type=%s severity=%s actor=%s",
                    change_type,
                    ContractSeverity.BLOCKING.value,
                    getattr(sc, 'actor', None)
                )

        passed = len(blocking_missing) == 0

        total_count = len(state_changes)
        if total_count == 0:
            overall_confidence = 1.0
            weight_applied = 1.0
        else:
            avg_confidence = sum(e.confidence for e in matched_evidences) / total_count if total_count > 0 else 0.0
            avg_weight = total_weighted / total_count if total_count > 0 else 0.0
            overall_confidence = avg_confidence
            weight_applied = avg_weight

        errors = []
        if blocking_missing:
            errors.append(f"Blocking missing: {', '.join(blocking_missing[:3])}")

        errors = []
        if blocking_missing:
            errors.append(f"Blocking missing: {', '.join(blocking_missing[:3])}")

        result = ValidationResult(
            passed=passed,
            missing=missing,
            matched=matched_evidences,
            blocking_missing=blocking_missing,
            overall_confidence=overall_confidence,
            weight_applied=weight_applied,
            errors=errors,
            missing_changes=missing_changes,   # 新增
        )

        logger.info(
            "SemanticValidator result: passed=%s, matched=%d, missing=%d, blocking=%d",
            result.passed,
            result.match_count,
            result.missing_count,
            result.blocking_missing_count
        )
        logger.critical("D5.1 ValidationResult missing_changes count=%d", len(missing_changes))
        return result


    def _run_pipeline(
        self,
        state_change: StateChange,
        event_text: str,
        scene_text: str,
        scene_id: str,
    ) -> tuple[bool, ValidationEvidence]:
        best_confidence = 0.0

        # Stage 1-3: 确定性匹配器
        for matcher in self._matchers:
            result = matcher.match(event_text, scene_text)
            if result.matched:
                return True, self._to_evidence(
                    state_change, event_text, scene_id, result, matcher.name
                )
            if result.confidence > best_confidence:
                best_confidence = result.confidence

        # Stage 4: Embedding (仅当最佳置信度 < min_confidence)
        if (
            self._enable_embedding
            and best_confidence < self._embedding_min_confidence
        ):
            embedding_result = self._embedding_matcher.match(event_text, scene_text)
            if embedding_result.matched:
                return True, self._to_evidence(
                    state_change, event_text, scene_id, embedding_result, "embedding"
                )
            if embedding_result.confidence > best_confidence:
                best_confidence = embedding_result.confidence

        fallback_evidence = ValidationEvidence(
            evidence_id=ValidationEvidence.generate_id(
                scene_id,
                state_change.id or event_text[:20],
                "none",
                "",
            ),
            event_id=state_change.id or "",
            event_text=event_text,
            matcher="none",
            confidence=best_confidence or 0.0,
            source=state_change.source,
            matched_text="",
            weight=self._weight_policy.weighted_score(0.0, state_change.source),
        )
        return False, fallback_evidence

    def _to_evidence(
        self,
        state_change: StateChange,
        event_text: str,
        scene_id: str,
        result: MatcherResult,
        matcher_name: str,
    ) -> ValidationEvidence:
        return ValidationEvidence(
            evidence_id=ValidationEvidence.generate_id(
                scene_id,
                state_change.id or event_text[:20],
                matcher_name,
                result.matched_text,
            ),
            event_id=state_change.id or "",
            event_text=event_text,
            matcher=matcher_name,
            confidence=result.confidence,
            source=state_change.source,
            matched_text=result.matched_text,
            weight=self._weight_policy.weighted_score(result.confidence, state_change.source),
        )

    def _get_event_text(self, state_change: StateChange) -> str:
        # 优先使用业务字段
        if state_change.name:
            return state_change.name
        if state_change.item:
            return state_change.item
        if state_change.location:
            return state_change.location
        if state_change.to_char:
            # 组合 from_char 和 to_char 用于匹配
            return f"{state_change.from_char}_{state_change.to_char}"
        # 最后回退到类型字符串（保留可观测性）
        if hasattr(state_change.type, "value"):
            return state_change.type.value
        return str(state_change.type)
    
    @staticmethod
    def _contract_type_value(state_change: StateChange) -> str:
        """安全获取 StateChange.type 的字符串值"""
        t = state_change.type
        if hasattr(t, "value"):
            return t.value
        return str(t)

    def _describe_state_change(self, state_change: StateChange) -> str:
        """生成自然语言描述，用于 FeedbackCompiler"""
        t = self._contract_type_value(state_change)
        if t == "plot_flag":
            return f"剧情标记 {state_change.name} 应设为 {state_change.value}"
        if t == "inventory_acquire":
            return f"{state_change.actor} 应获得 {state_change.item}"
        if t == "location_change":
            return f"{state_change.actor} 应到达 {state_change.location}"
        if t == "realm_change":
            return f"{state_change.actor} 应突破至 {state_change.to_major_realm} {state_change.to_minor_stage}层"
        if t == "relationship_change":
            return f"{state_change.from_char} 与 {state_change.to_char} 的关系应变化 {state_change.delta}"
        if t == "knowledge_gain":
            return f"应获得知识：{state_change.name}"
        return f"缺失类型 {t} 的状态变化"