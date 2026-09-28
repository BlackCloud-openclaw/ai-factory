"""
C3.4.3B.2: Bridge Outcome Audit Writer

将 B2-2 bridge 的调用结果写入 bridge_outcome_audit 表。

设计原则：
- 纯写入，不改变 bridge 判定语义
- 失败不传播，只记录日志
- 单条写入返回 bool，语义真实反映 INSERT 结果
- 与 audit_writer.py 同构
- bounded timeout 由调用方（bridge）包装，本模块不引入时间依赖
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional, List

import asyncpg

logger = logging.getLogger(__name__)


BRIDGE_VERSION = "b2_2c2.v1"


@dataclass(frozen=True)
class BridgeAuditContext:
    """
    bridge outcome 写入的上下文。

    与 AuditContext 独立：这里没有 source / mode / contract_hash，
    因为 bridge 层不区分 production / shadow（那是 claim-level 的概念）。
    """
    novel_id: str
    volume_num: Optional[int] = None
    chapter_num: Optional[int] = None
    scene_idx: Optional[int] = None
    scene_id: Optional[str] = None


async def record_bridge_outcome(
    conn: asyncpg.Connection,
    context: BridgeAuditContext,
    claim_types: List[str],
    production_claim_types: List[str],
    trigger_status: str,
    triggered: bool,
    execution_status: str = "completed",
    error_reason: Optional[str] = None,
    all_supported: Optional[bool] = None,
    min_confidence: Optional[float] = None,
    confidence_threshold: Optional[float] = None,
    confidence_ok: Optional[bool] = None,
    rescued: Optional[bool] = None,
    representative_claim_id: Optional[str] = None,
    representative_claim_type: Optional[str] = None,
    representative_reason: Optional[str] = None,
    representative_evidence_ids: Optional[List[str]] = None,
) -> bool:
    """
    写入单条 bridge outcome 记录。

    Returns:
        True  → INSERT 成功
        False → INSERT 失败（异常已记录日志，不向上传播）
    """
    try:
        await conn.execute(
            """
            INSERT INTO bridge_outcome_audit (
                novel_id, volume_num, chapter_num, scene_idx, scene_id,
                claim_types, production_claim_types,
                trigger_status, triggered,
                execution_status, error_reason,
                all_supported, min_confidence, confidence_threshold,
                confidence_ok, rescued,
                representative_claim_id, representative_claim_type,
                representative_reason, representative_evidence_ids,
                bridge_version
            ) VALUES (
                $1,  $2,  $3,  $4,  $5,
                $6,  $7,
                $8,  $9,
                $10, $11,
                $12, $13, $14,
                $15, $16,
                $17, $18,
                $19, $20,
                $21
            )
            """,
            context.novel_id,
            context.volume_num,
            context.chapter_num,
            context.scene_idx,
            context.scene_id,
            json.dumps(claim_types, ensure_ascii=False),
            json.dumps(production_claim_types, ensure_ascii=False),
            trigger_status,
            triggered,
            execution_status,
            error_reason,
            all_supported,
            min_confidence,
            confidence_threshold,
            confidence_ok,
            rescued,
            representative_claim_id,
            representative_claim_type,
            representative_reason,
            (
                json.dumps(representative_evidence_ids, ensure_ascii=False)
                if representative_evidence_ids is not None
                else None
            ),
            BRIDGE_VERSION,
        )
        return True
    except Exception as e:
        logger.error(
            f"[BridgeAuditWriter] Failed to write: "
            f"scene_id={context.scene_id}, trigger_status={trigger_status}, "
            f"error={type(e).__name__}: {e}",
            exc_info=True,
        )
        return False