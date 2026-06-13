"""DeltaEngine - 纯函数，计算 PredicateDelta"""
import logging
import re
from typing import Dict, List, Optional, Any
from src.writing.causality.predicate import Predicate
from src.writing.causality.delta import PredicateDelta, PredicateRef
from src.writing.causality.projection_store import ProjectionStore
from src.writing.causality.health import HealthChecker
from src.db import get_db_pool

logger = logging.getLogger(__name__)


class DeltaEngine:
    """将事件转换为 PredicateDelta（纯函数，无副作用）"""
    # 核心单值关系（由 ADR-006 定义）
    SINGLETON_RELATIONS = {"realm", "is_alive", "location"}

    @staticmethod
    def _normalize_realm(realm_str: str) -> str:
        if not realm_str:
            return realm_str
        cleaned = re.sub(r'[一二三四五六七八九零\d]+(?:层|期|重|级)?$', '', realm_str)
        if not cleaned:
            return realm_str
        return cleaned.strip()

    def compute_delta(
        self,
        current_active: Dict[str, Predicate],  # key = identity_key
        event: Dict[str, Any]
    ) -> PredicateDelta:
        novel_id = event.get('novel_id')
        event_id = event.get('id') or event.get('event_id')
        if not novel_id or not event_id:
            raise ValueError("Event missing novel_id or event_id")

        semantic = event.get('semantic', 'state_mutation')
        event_type = event.get('type', '')

        if semantic in ('dream', 'illusion', 'flashback'):
            base_confidence = 0.4
            base_priority = 'flavor'
        elif semantic in ('dialogue', 'observation'):
            return PredicateDelta(
                novel_id=novel_id,
                event_id=event_id,
                projection_version=1,
                event_semantic=semantic,
                to_activate=[],
                to_deactivate=[]
            )
        else:
            base_confidence = 1.0
            base_priority = 'core' if event_type in self._core_event_types() else 'narrative'

        to_activate = []
        to_deactivate = []

        # 处理角色首次出现（NPC 引入）
        if event_type in ('npc_introduce', 'character_appear'):
            actor = event.get('name') or event.get('actor')
            if actor:
                # 激活 is_alive 谓词
                alive_pred = Predicate(
                    subject=actor,
                    relation='is_alive',
                    object=True,
                    confidence=base_confidence,
                    priority='core',
                    source_event_id=event_id,
                    source_event_type=event_type,
                    source_event_semantic=semantic
                )
                to_activate.append(alive_pred)
                logger.debug(f"Activated is_alive for {actor}")

        # 处理物品获取
        if event_type == 'item_acquire':
            actor = event.get('actor')
            item = event.get('item')
            if actor and item:
                pred = Predicate(
                    subject=actor,
                    relation='has_item',
                    object=item,
                    confidence=base_confidence,
                    priority=base_priority,
                    source_event_id=event_id,
                    source_event_type=event_type,
                    source_event_semantic=semantic
                )
                to_activate.append(pred)

        # 处理物品丢失
        elif event_type == 'item_lose':
            actor = event.get('actor')
            item = event.get('item')
            if actor and item:
                target_identity = Predicate(
                    subject=actor,
                    relation='has_item',
                    object=item
                ).identity_key()
                if target_identity in current_active:
                    to_deactivate.append(PredicateRef(
                        identity_key=target_identity,
                        event_id=event_id
                    ))

        # 处理境界突破
        elif event_type == 'realm_upgrade':
            actor = event.get('actor')
            to_major_realm = event.get('to_major_realm')
            if actor and to_major_realm:
                if hasattr(to_major_realm, 'value'):
                    realm_str = to_major_realm.value
                else:
                    realm_str = to_major_realm
                normalized_realm = self._normalize_realm(realm_str)
                new_pred = Predicate(
                    subject=actor,
                    relation='realm',
                    object=normalized_realm,
                    confidence=base_confidence,
                    priority='core',
                    source_event_id=event_id,
                    source_event_type=event_type,
                    source_event_semantic=semantic
                )
                to_activate.append(new_pred)

        # 处理位置进入（可选，激活 location 谓词）
        elif event_type == 'location_enter':
            actor = event.get('actor')
            location = event.get('location')
            if actor and location:
                loc_pred = Predicate(
                    subject=actor,
                    relation='location',
                    object=location,
                    confidence=base_confidence,
                    priority='core',
                    source_event_id=event_id,
                    source_event_type=event_type,
                    source_event_semantic=semantic
                )
                to_activate.append(loc_pred)

        return PredicateDelta(
            novel_id=novel_id,
            event_id=event_id,
            projection_version=1,
            event_semantic=semantic,
            to_activate=to_activate,
            to_deactivate=to_deactivate
        )

    def _core_event_types(self) -> set:
        return {'realm_upgrade', 'item_acquire', 'relationship_change', 'location_enter', 'npc_introduce', 'character_appear'}
    
    @staticmethod
    async def rebuild_all_predicates(novel_id: str, pool=None):
        from src.writing.event_store import NarrativeEventStore

        pool = pool or get_db_pool()
        if not pool:
            raise RuntimeError("Database pool not available")

        event_store = NarrativeEventStore(pool)
        store = ProjectionStore(pool)

        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM predicates WHERE novel_id = $1", novel_id)
            await conn.execute("DELETE FROM projection_applied WHERE novel_id = $1", novel_id)

        events = await event_store.get_events_since(novel_id, since_event_id=0, limit=1000000)
        current_active: Dict[str, Predicate] = {}
        delta_engine = DeltaEngine()

        for evt_id, evt in events:
            event_dict = evt.model_dump()
            event_dict["id"] = evt_id
            event_dict["novel_id"] = novel_id
            if "semantic" not in event_dict:
                event_dict["semantic"] = "state_mutation"

            delta = delta_engine.compute_delta(current_active, event_dict)
            await store.apply_delta(delta)

            for pred in delta.to_activate:
                current_active[pred.identity_key()] = pred
            for ref in delta.to_deactivate:
                current_active.pop(ref.identity_key, None)

        async with pool.acquire() as conn:
            core_hash = await HealthChecker._compute_core_predicates_hash(novel_id)
            await conn.execute(
                """
                INSERT INTO projection_health (novel_id, last_full_rebuild_at, core_predicates_hash, updated_at)
                VALUES ($1, NOW(), $2, NOW())
                ON CONFLICT (novel_id) DO UPDATE
                SET last_full_rebuild_at = NOW(), core_predicates_hash = $2, updated_at = NOW()
                """,
                novel_id, core_hash
            )
        logger.info(f"Full rebuild of predicates completed for novel {novel_id}, processed {len(events)} events")

    @staticmethod
    async def verify_consistency(novel_id: str, auto_repair: bool = False) -> Dict[str, Any]:
        pool = get_db_pool()
        if not pool:
            raise RuntimeError("Database pool not available")
        
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT core_predicates_hash, last_full_rebuild_at FROM projection_health WHERE novel_id = $1",
                novel_id
            )
        current_hash = row["core_predicates_hash"] if row else None
        
        if auto_repair:
            await DeltaEngine.rebuild_all_predicates(novel_id, pool)
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT core_predicates_hash FROM projection_health WHERE novel_id = $1",
                    novel_id
                )
            new_hash = row["core_predicates_hash"] if row else None
            consistent = (current_hash == new_hash) if current_hash else (new_hash is not None)
            return {
                "consistent": consistent,
                "current_hash": current_hash,
                "rebuilt_hash": new_hash,
                "auto_repaired": True,
            }
        else:
            return {
                "consistent": None,
                "current_hash": current_hash,
                "rebuilt_hash": None,
                "auto_repaired": False,
                "message": "Run with auto_repair=True to perform full rebuild and compare."
            }
