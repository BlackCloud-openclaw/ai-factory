"""
B2-2C2: B2-2 Production Bridge (Frozen, Revised)

Phase 15.7-C2 冻结配置：
- 白名单：["plot_flag"]
- Shadow-only：realm_change, knowledge_gain, relationship_change,
                inventory_acquire, location_change
- confidence 阈值：0.90
- multi-claim 安全：使用 min(confidence) 而非 max

C3.4.2d（claim-level audit）:
- try_rescue 增加可选 audit_context 参数

C3.4.3B.3（bridge-level audit）:
- try_rescue 增加可选 bridge_audit_context 参数
- 每个 return 分支都写入 bridge_outcome_audit

Audit budget 语义（C3.4.3B.3 修正）:
- 1 秒是 "audit persistence 自身" 的总预算，
  不包含 validate_contract() 内部的 LLM / SemanticJudge 耗时。
- triggered 路径：deadline 在 validate_contract() 返回后建立，
  覆盖 claim audit + bridge audit。
- skip 分支：各自建立独立 deadline（无长耗时工作）。
- 异常分支：建立新 deadline（异常路径可能发生在 deadline 之前）。
- 每个 audit 使用 deadline - now 作为剩余预算；
  预算耗尽则跳过并记录日志。
- 判定逻辑零改动。
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from .validator import ValidationV2
from .models import Verdict, MatchLayer
from .audit_writer import AuditContext, record_audit_batch
from .bridge_audit_writer import BridgeAuditContext, record_bridge_outcome

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class B2_2ProductionResult:
    """B2-2 生产判断结果"""
    triggered: bool
    verdict: Optional[str] = None
    confidence: float = 0.0
    matched_layer: Optional[str] = None
    reason: str = ""
    claim_id: str = ""
    claim_type: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    rescued: bool = False


class B2_2ProductionBridge:
    """
    B2-2 生产桥接层 (Phase 15.7-C2 Revised, Frozen)
    """

    PRODUCTION_TYPES: List[str] = ["plot_flag"]

    SHADOW_ONLY_TYPES: List[str] = [
        "realm_change",
        "knowledge_gain",
        "relationship_change",
        "inventory_acquire",
        "location_change",
    ]

    CONFIDENCE_THRESHOLD: float = 0.90

    # audit persistence 总预算（秒）
    # 只覆盖 claim audit + bridge audit，不包含 validate_contract
    AUDIT_WRITE_TIMEOUT: float = 1.0

    def __init__(
        self,
        confidence_threshold: float = None,
        production_types: List[str] = None,
        audit_write_timeout: float = None,
    ):
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.CONFIDENCE_THRESHOLD
        )
        self.production_types = (
            production_types
            if production_types is not None
            else list(self.PRODUCTION_TYPES)
        )
        self._audit_write_timeout = (
            audit_write_timeout
            if audit_write_timeout is not None
            else self.AUDIT_WRITE_TIMEOUT
        )
        self._validator = ValidationV2()
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """启用/禁用 B2-2 生产桥接（kill switch）"""
        self._enabled = enabled

    def _should_apply(self, claim_type: str) -> bool:
        """唯一白名单入口。"""
        if not self._enabled:
            return False
        if claim_type in self.SHADOW_ONLY_TYPES:
            return False
        if claim_type not in self.production_types:
            return False
        return True

    def _new_audit_deadline(self) -> float:
        """
        新建 audit deadline。

        语义：本 deadline 仅覆盖 audit persistence；
        调用者需确保在 deadline 建立后不再执行长耗时业务逻辑。
        """
        return asyncio.get_running_loop().time() + self._audit_write_timeout

    # ============================================================
    # Audit writers (bounded, non-blocking)
    # ============================================================

    async def _write_claim_audit_bounded(
        self,
        audit_context: Optional[AuditContext],
        results: List[Any],
        deadline: float,
    ) -> None:
        """bounded claim-level audit。使用 deadline - now 作为剩余预算。"""
        if audit_context is None:
            return

        loop = asyncio.get_running_loop()
        remaining = deadline - loop.time()
        if remaining <= 0:
            logger.warning(
                "[B2-2C2] Claim audit budget exhausted, skipping"
            )
            return

        try:
            await asyncio.wait_for(
                self._write_claim_audit_impl(results, audit_context),
                timeout=remaining,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[B2-2C2] Claim audit write timeout "
                f"(> {remaining:.2f}s remaining budget)"
            )
        except Exception as e:
            logger.warning(
                f"[B2-2C2] Claim audit write failed (non-blocking): "
                f"{type(e).__name__}: {e}"
            )

    async def _write_claim_audit_impl(
        self,
        results: List[Any],
        audit_context: AuditContext,
    ) -> None:
        """实际写入 claim-level audit（拆分 production / shadow_only）。"""
        from src.db import get_db_pool

        pool = get_db_pool()
        if pool is None:
            logger.debug("[B2-2C2] Claim audit skipped: db pool unavailable")
            return

        production = [
            r for r in results
            if self._should_apply(r.state_change_type)
        ]
        shadow_only = [
            r for r in results
            if (not self._should_apply(r.state_change_type))
            and r.state_change_type in self.SHADOW_ONLY_TYPES
        ]

        async with pool.acquire() as conn:
            if production:
                prod_ctx = AuditContext(
                    novel_id=audit_context.novel_id,
                    volume_num=audit_context.volume_num,
                    chapter_num=audit_context.chapter_num,
                    scene_idx=audit_context.scene_idx,
                    scene_id=audit_context.scene_id,
                    source=audit_context.source,
                    mode="production",
                    contract_hash=audit_context.contract_hash,
                )
                n = await record_audit_batch(conn, production, prod_ctx)
                logger.info(
                    f"[B2-2C2] Claim audit (production): "
                    f"{n}/{len(production)}"
                )

            if shadow_only:
                shadow_ctx = AuditContext(
                    novel_id=audit_context.novel_id,
                    volume_num=audit_context.volume_num,
                    chapter_num=audit_context.chapter_num,
                    scene_idx=audit_context.scene_idx,
                    scene_id=audit_context.scene_id,
                    source=audit_context.source,
                    mode="shadow_only",
                    contract_hash=audit_context.contract_hash,
                )
                n = await record_audit_batch(conn, shadow_only, shadow_ctx)
                logger.info(
                    f"[B2-2C2] Claim audit (shadow_only): "
                    f"{n}/{len(shadow_only)}"
                )

    async def _write_bridge_audit_bounded(
        self,
        bridge_audit_context: Optional[BridgeAuditContext],
        deadline: float,
        claim_types: List[str],
        production_claim_types: List[str],
        trigger_status: str,
        triggered: bool,
        execution_status: str = "completed",
        error_reason: Optional[str] = None,
        all_supported: Optional[bool] = None,
        min_confidence: Optional[float] = None,
        confidence_ok: Optional[bool] = None,
        rescued: Optional[bool] = None,
        representative_claim_id: Optional[str] = None,
        representative_claim_type: Optional[str] = None,
        representative_reason: Optional[str] = None,
        representative_evidence_ids: Optional[List[str]] = None,
    ) -> None:
        """bounded bridge-level audit。使用 deadline - now 作为剩余预算。"""
        if bridge_audit_context is None:
            return

        loop = asyncio.get_running_loop()
        remaining = deadline - loop.time()
        if remaining <= 0:
            logger.warning(
                f"[B2-2C2] Bridge audit budget exhausted, skipping "
                f"(trigger_status={trigger_status})"
            )
            return

        try:
            await asyncio.wait_for(
                self._write_bridge_audit_impl(
                    bridge_audit_context=bridge_audit_context,
                    claim_types=claim_types,
                    production_claim_types=production_claim_types,
                    trigger_status=trigger_status,
                    triggered=triggered,
                    execution_status=execution_status,
                    error_reason=error_reason,
                    all_supported=all_supported,
                    min_confidence=min_confidence,
                    confidence_ok=confidence_ok,
                    rescued=rescued,
                    representative_claim_id=representative_claim_id,
                    representative_claim_type=representative_claim_type,
                    representative_reason=representative_reason,
                    representative_evidence_ids=representative_evidence_ids,
                ),
                timeout=remaining,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[B2-2C2] Bridge audit write timeout "
                f"(> {remaining:.2f}s remaining budget, "
                f"trigger_status={trigger_status})"
            )
        except Exception as e:
            logger.warning(
                f"[B2-2C2] Bridge audit write failed (non-blocking): "
                f"{type(e).__name__}: {e}"
            )

    async def _write_bridge_audit_impl(
        self,
        bridge_audit_context: BridgeAuditContext,
        claim_types: List[str],
        production_claim_types: List[str],
        trigger_status: str,
        triggered: bool,
        execution_status: str,
        error_reason: Optional[str],
        all_supported: Optional[bool],
        min_confidence: Optional[float],
        confidence_ok: Optional[bool],
        rescued: Optional[bool],
        representative_claim_id: Optional[str],
        representative_claim_type: Optional[str],
        representative_reason: Optional[str],
        representative_evidence_ids: Optional[List[str]],
    ) -> None:
        """实际写入 bridge audit 表。"""
        from src.db import get_db_pool

        pool = get_db_pool()
        if pool is None:
            logger.debug("[B2-2C2] Bridge audit skipped: db pool unavailable")
            return

        async with pool.acquire() as conn:
            ok = await record_bridge_outcome(
                conn=conn,
                context=bridge_audit_context,
                claim_types=claim_types,
                production_claim_types=production_claim_types,
                trigger_status=trigger_status,
                triggered=triggered,
                execution_status=execution_status,
                error_reason=error_reason,
                all_supported=all_supported,
                min_confidence=min_confidence,
                confidence_threshold=self.confidence_threshold,
                confidence_ok=confidence_ok,
                rescued=rescued,
                representative_claim_id=representative_claim_id,
                representative_claim_type=representative_claim_type,
                representative_reason=representative_reason,
                representative_evidence_ids=representative_evidence_ids,
            )
            if ok:
                logger.info(
                    f"[B2-2C2] Bridge audit written: "
                    f"trigger_status={trigger_status}, "
                    f"execution_status={execution_status}, "
                    f"triggered={triggered}, rescued={rescued}"
                )

    # ============================================================
    # 主流程
    # ============================================================

    async def try_rescue(
        self,
        contract: Dict[str, Any],
        writer_events: List[Dict[str, Any]],
        scene_text: str,
        original_validation_result: Dict[str, Any],
        scene_id: str,
        audit_context: Optional[AuditContext] = None,
        bridge_audit_context: Optional[BridgeAuditContext] = None,
    ) -> B2_2ProductionResult:
        """
        尝试用 B2-2 救回原本 FAIL 的验证。

        Audit budget 边界：
        - triggered 路径：deadline 在 validate_contract() 返回后建立，
          只覆盖 claim audit + bridge audit。
        - skip 分支：各自建立独立 deadline（无长耗时工作）。
        - 异常分支：建立新 deadline。
        """

        # ---------- 1. original_passed（skip 分支）----------
        if original_validation_result.get("passed", False):
            result = B2_2ProductionResult(
                triggered=False,
                verdict="SKIPPED",
                reason="Original validator already passed",
            )
            await self._write_bridge_audit_bounded(
                bridge_audit_context=bridge_audit_context,
                deadline=self._new_audit_deadline(),
                claim_types=[],
                production_claim_types=[],
                trigger_status="original_passed",
                triggered=False,
                execution_status="completed",
            )
            return result

        # ---------- 2. no_state_changes（skip 分支）----------
        state_changes = contract.get("observables", {}).get("state_changes", [])
        if not state_changes:
            result = B2_2ProductionResult(
                triggered=False,
                verdict="SKIPPED",
                reason="No state_changes in contract",
            )
            await self._write_bridge_audit_bounded(
                bridge_audit_context=bridge_audit_context,
                deadline=self._new_audit_deadline(),
                claim_types=[],
                production_claim_types=[],
                trigger_status="no_state_changes",
                triggered=False,
                execution_status="completed",
            )
            return result

        claim_types = [sc.get("type", "unknown") for sc in state_changes]

        # ---------- 3. no_production_types（skip 分支）----------
        production_types_in_contract = [
            t for t in claim_types if self._should_apply(t)
        ]

        if not production_types_in_contract:
            result = B2_2ProductionResult(
                triggered=False,
                verdict="SKIPPED",
                reason=f"No production types in contract (types={claim_types})",
            )
            await self._write_bridge_audit_bounded(
                bridge_audit_context=bridge_audit_context,
                deadline=self._new_audit_deadline(),
                claim_types=claim_types,
                production_claim_types=[],
                trigger_status="no_production_types",
                triggered=False,
                execution_status="completed",
            )
            return result

        # ---------- 4. 进入 rescue 逻辑 ----------
        try:
            # -------- 关键：validate_contract 不计入 audit budget --------
            results = await self._validator.validate_contract(
                contract=contract,
                writer_events=writer_events or [],
                scene_text=scene_text or "",
            )

            # ============================================================
            # C3.4.3B.3 修正：audit deadline 现在才建立
            #
            # 语义：1 秒预算仅覆盖 audit persistence，
            # 不包含 validate_contract() 内部的 LLM / SemanticJudge 耗时。
            # ============================================================
            audit_deadline = self._new_audit_deadline()
            # ============================================================

            # -------- claim-level audit（共享 deadline） --------
            await self._write_claim_audit_bounded(
                audit_context=audit_context,
                results=results,
                deadline=audit_deadline,
            )

            # -------- 生产过滤 --------
            production_results = [
                r for r in results
                if self._should_apply(r.state_change_type)
            ]

            # -------- 4a: 无 production results --------
            if not production_results:
                result = B2_2ProductionResult(
                    triggered=True,
                    verdict="INSUFFICIENT",
                    reason="No production-type results from validator",
                    rescued=False,
                )
                await self._write_bridge_audit_bounded(
                    bridge_audit_context=bridge_audit_context,
                    deadline=audit_deadline,
                    claim_types=claim_types,
                    production_claim_types=production_types_in_contract,
                    trigger_status="triggered",
                    triggered=True,
                    execution_status="completed",
                    all_supported=False,
                    min_confidence=0.0,
                    confidence_ok=False,
                    rescued=False,
                    representative_reason=(
                        "No production-type results from validator"
                    ),
                )
                return result

            # -------- 4b: 正常计算 rescue 判定 --------
            all_supported = all(
                r.verdict == Verdict.SUPPORTED
                for r in production_results
            )
            min_confidence = min(
                (r.confidence for r in production_results),
                default=0.0,
            )
            confidence_ok = min_confidence >= self.confidence_threshold
            rescued = all_supported and confidence_ok

            evidence_ids: List[str] = []
            for r in production_results:
                if r.judgement is not None:
                    for eid in (r.judgement.evidence_ids or []):
                        if eid not in evidence_ids:
                            evidence_ids.append(eid)

            representative = next(
                (r for r in production_results if r.verdict == Verdict.SUPPORTED),
                production_results[0],
            )

            logger.info(
                f"[B2-2C2] scene={scene_id}, "
                f"production_types={production_types_in_contract}, "
                f"claims={len(production_results)}, "
                f"all_supported={all_supported}, "
                f"min_confidence={min_confidence:.2f}, "
                f"threshold={self.confidence_threshold}, "
                f"rescued={rescued}"
            )

            result = B2_2ProductionResult(
                triggered=True,
                verdict="SUPPORTED" if all_supported else "INSUFFICIENT",
                confidence=min_confidence,
                matched_layer="semantic",
                reason=representative.reason,
                claim_id=representative.claim_id,
                claim_type=representative.state_change_type,
                evidence_ids=evidence_ids,
                rescued=rescued,
            )

            await self._write_bridge_audit_bounded(
                bridge_audit_context=bridge_audit_context,
                deadline=audit_deadline,
                claim_types=claim_types,
                production_claim_types=production_types_in_contract,
                trigger_status="triggered",
                triggered=True,
                execution_status="completed",
                all_supported=all_supported,
                min_confidence=min_confidence,
                confidence_ok=confidence_ok,
                rescued=rescued,
                representative_claim_id=representative.claim_id,
                representative_claim_type=representative.state_change_type,
                representative_reason=representative.reason,
                representative_evidence_ids=evidence_ids,
            )
            return result

        # ---------- 5. 异常分支 ----------
        except Exception as e:
            logger.error(
                f"[B2-2C2] Validation failed for {scene_id}: {e}",
                exc_info=True,
            )
            result = B2_2ProductionResult(
                triggered=True,
                verdict="ERROR",
                confidence=0.0,
                reason=f"B2-2 error: {e}",
                rescued=False,
            )
            # 异常分支建立新 deadline（异常可能发生在 validate_contract 内）
            await self._write_bridge_audit_bounded(
                bridge_audit_context=bridge_audit_context,
                deadline=self._new_audit_deadline(),
                claim_types=claim_types,
                production_claim_types=production_types_in_contract,
                trigger_status="triggered",
                triggered=True,
                execution_status="error",
                error_reason=f"{type(e).__name__}: {e}",
            )
            return result


# ============================================================
# 全局实例
# ============================================================
_bridge: Optional[B2_2ProductionBridge] = None


def get_b2_2_bridge() -> B2_2ProductionBridge:
    """获取 B2-2 生产桥接实例"""
    global _bridge
    if _bridge is None:
        _bridge = B2_2ProductionBridge()
    return _bridge