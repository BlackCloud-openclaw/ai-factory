from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
from src.db import get_db_pool
import json

# 定义指标
llm_request_duration = Histogram(
    'ai_factory_llm_request_duration_seconds',
    'LLM request duration by agent',
    ['agent', 'model']
)
llm_requests_total = Counter(
    'ai_factory_llm_requests_total',
    'Total LLM requests',
    ['agent', 'model', 'status']
)
projection_lag = Gauge('ai_factory_projection_lag_events', 'Number of events not yet projected')
entropy_local = Gauge('ai_factory_entropy_local', 'Local narrative entropy')
entropy_arc = Gauge('ai_factory_entropy_arc', 'Arc entropy')
entropy_civ = Gauge('ai_factory_entropy_civilization', 'Civilization entropy')
active_predicates = Gauge('ai_factory_active_predicates', 'Number of active predicates')
dead_letter_count = Gauge('ai_factory_dead_letter_count', 'Number of dead letters')

async def metrics_endpoint():
    """Prometheus metrics 端点"""
    pool = get_db_pool()
    if pool:
        async with pool.acquire() as conn:
            # 投影滞后
            row = await conn.fetchrow(
                "SELECT last_projected_event_id FROM projection_health WHERE novel_id = 'simple_long_novel_001'"
            )
            if row:
                last_event = await conn.fetchval("SELECT MAX(id) FROM narrative_events WHERE novel_id = 'simple_long_novel_001'")
                if last_event and row["last_projected_event_id"]:
                    projection_lag.set(last_event - row["last_projected_event_id"])
            # 死信数量
            dead = await conn.fetchval("SELECT COUNT(*) FROM projection_dead_letters WHERE novel_id = 'simple_long_novel_001'")
            dead_letter_count.set(dead or 0)
            # 活跃谓词数量
            active = await conn.fetchval("SELECT COUNT(*) FROM predicates WHERE novel_id = 'simple_long_novel_001' AND is_active = true")
            active_predicates.set(active or 0)
            # 熵值（从最新快照）
            row = await conn.fetchrow(
                "SELECT compressed_state FROM world_snapshots WHERE novel_id = 'simple_long_novel_001' ORDER BY snapshot_id DESC LIMIT 1"
            )
            if row and row["compressed_state"]:
                comp = json.loads(row["compressed_state"])
                entropy_local.set(comp.get("local_entropy", 0))
                entropy_arc.set(comp.get("arc_entropy", 0))
                entropy_civ.set(comp.get("civilization_entropy", 0))
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)