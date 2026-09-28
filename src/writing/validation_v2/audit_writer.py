"""
C3.4.2c: Validator V2 Audit Writer

将 ValidationResultV2 写入 validator_v2_audit 表。

设计原则：
- 纯写入，不改变任何判定语义
- 失败不传播，只记录日志（不阻塞生产）
- 支持单条与批量写入
- 三态语义（None / [] / [N]）严格保留
- 单条 writer 返回 bool，让 batch 计数的 success 具有真实语义

使用方式：
    conn: asyncpg.Connection（由调用方提供，本模块不创建）
    context = AuditContext(
        novel_id="simple_long_novel_001",
        volume_num=1, chapter_num=29, scene_idx=0,
        scene_id="scene_1_29_0",
        source="production", mode="production",
    )
    written = await record_audit_batch(conn, results, context)
    # written 表示真正成功写入的条数
"""

import json
import logging
from typing import Optional, List, Sequence
from dataclasses import dataclass

import asyncpg

from .models import ValidationResultV2


logger = logging.getLogger(__name__)


VALIDATOR_VERSION = "v2.0"


@dataclass(frozen=True)
class AuditContext:
    """
    audit 写入的上下文（与具体 claim 无关）。

    source:
        - production:    来自 validate_node 的正常调用
        - c3_3b_eval:    来自离线评测脚本
        - b2_2_bridge:   来自 B2-2 救援入口

    mode:
        - production:    落在 B2-2 白名单内的 claim
        - shadow_only:   白名单外但仍记录的 claim
    """
    novel_id: str
    volume_num: Optional[int] = None
    chapter_num: Optional[int] = None
    scene_idx: Optional[int] = None
    scene_id: Optional[str] = None
    source: str = "production"
    mode: str = "production"
    contract_hash: Optional[str] = None


def _to_json_or_none(value: Optional[List[str]]) -> Optional[str]:
    """
    三态语义保留：
        None  → None（Retriever 未调用）
        []    → "[]"（Retriever 已调用，未命中）
        [...] → "[...]"（命中 N 条）
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


async def record_audit_result(
    conn: asyncpg.Connection,
    result: ValidationResultV2,
    context: AuditContext,
) -> bool:
    """
    写入单条 audit 记录。

    Returns:
        True  → INSERT 成功
        False → INSERT 失败（异常已记录日志，不向上传播）

    调用方负责决定是否将失败视为阻塞；本模块永远不抛出。
    """
    try:
        # ---- 三态语义保留 ----
        retrieved_ids_json = _to_json_or_none(result.retrieved_evidence_ids)

        # ---- LLM 证据 ----
        if result.judgement is not None:
            llm_evidence_ids_json = json.dumps(
                result.judgement.evidence_ids or [],
                ensure_ascii=False,
            )
            llm_raw_verdict = result.judgement.verdict.value
            llm_raw_confidence = result.judgement.confidence
            llm_raw_reason = result.judgement.reason
        else:
            llm_evidence_ids_json = None
            llm_raw_verdict = None
            llm_raw_confidence = None
            llm_raw_reason = None

        await conn.execute(
            """
            INSERT INTO validator_v2_audit (
                novel_id, volume_num, chapter_num, scene_idx, scene_id,
                source, mode,
                claim_id, state_change_type, raw_state_change_type,
                structural_check,
                evidence_candidates_found,
                retrieved_evidence_count, retrieved_evidence_ids,
                llm_invoked,
                llm_raw_verdict, llm_raw_confidence, llm_raw_reason, llm_evidence_ids,
                final_verdict, matched_layer, final_confidence,
                fallback_applied, final_reason,
                validator_version, contract_hash
            ) VALUES (
                $1,  $2,  $3,  $4,  $5,
                $6,  $7,
                $8,  $9,  $10,
                $11,
                $12,
                $13, $14,
                $15,
                $16, $17, $18, $19,
                $20, $21, $22,
                $23, $24,
                $25, $26
            )
            """,
            context.novel_id,
            context.volume_num,
            context.chapter_num,
            context.scene_idx,
            context.scene_id,
            context.source,
            context.mode,
            result.claim_id,
            result.state_change_type,
            None,                          # raw_state_change_type（ValidationResultV2 暂未携带）
            result.structural_check,
            result.evidence_candidates_found,
            result.retrieved_evidence_count,
            retrieved_ids_json,
            result.llm_invoked,
            llm_raw_verdict,
            llm_raw_confidence,
            llm_raw_reason,
            llm_evidence_ids_json,
            result.verdict.value,
            result.matched_layer.value,
            result.confidence,
            result.fallback_applied,
            result.reason,
            VALIDATOR_VERSION,
            context.contract_hash,
        )
        return True

    except Exception as e:
        logger.error(
            "[AuditWriter] Failed to write audit: "
            f"claim_id={result.claim_id}, error={type(e).__name__}: {e}",
            exc_info=True,
        )
        return False


async def record_audit_batch(
    conn: asyncpg.Connection,
    results: Sequence[ValidationResultV2],
    context: AuditContext,
) -> int:
    """
    批量写入。逐条写入，单条失败不影响其他。

    Returns:
        真正 INSERT 成功的条数。
        语义：
            - 若所有 INSERT 均成功，返回值 == len(results)
            - 若有失败，返回值 < len(results)

        调用方可据此判断是否需要补写或告警。
    """
    if not results:
        return 0

    success = 0
    for r in results:
        if await record_audit_result(conn, r, context):
            success += 1
    return success