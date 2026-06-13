# src/orchestrator/audit.py
import logging
from typing import Optional
from src.db import get_db_pool
from src.orchestrator.state import AgentState

logger = logging.getLogger(__name__)


async def audit_state(state: AgentState, node_name: str):
    """
    审计当前状态，将状态哈希和最后事件ID存入数据库。
    应在每个工作流节点执行完毕后调用。
    """
    if not state.novel_id:
        logger.debug(f"Skipping audit: no novel_id in state (node={node_name})")
        return

    pool = get_db_pool()
    if not pool:
        logger.warning("Database pool not available, skipping state audit")
        return

    # 计算状态哈希
    state_hash = state.compute_state_hash()
    last_event_id = state.last_sequence_id or 0

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO state_audit (novel_id, node_name, step_count, last_event_id, state_hash, created_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            """,
            state.novel_id,
            node_name,
            state.step_count,
            last_event_id,
            state_hash
        )
    logger.debug(f"Audited state after {node_name}: hash={state_hash[:8]}..., last_event_id={last_event_id}")