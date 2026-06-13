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
        """
        将境界字符串规范化为大境界名称（去除数字和'层'字）
        例如: "炼气一层" -> "炼气", "金丹初期" -> "金丹", "元婴" -> "元婴"
        """
        if not realm_str:
            return realm_str
        # 移除数字（中文或阿拉伯）和"层"、"期"等后缀
        # 先尝试去除常见后缀
        cleaned = re.sub(r'[一二三四五六七八九零\d]+(?:层|期|重|级)?$', '', realm_str)
        # 如果结果为空，说明全是数字和层，返回原字符串（防御）
        if not cleaned:
            return realm_str
        return cleaned.strip()

    def compute_delta(
        self,
        current_active: Dict[str, Predicate],  # key = identity_key
        event: Dict[str, Any]
    ) -> PredicateDelta:
        """
        根据当前活跃谓词和事件，计算 Delta。
        输入：current_active 是内存字典（由调用方从数据库加载）。
        输出：PredicateDelta。
        此函数必须纯确定性，不访问数据库、不调用随机函数、不读取系统时间。
        """
        novel_id = event.get('novel_id')
        event_id = event.get('id') or event.get('event_id')
        if not novel_id or not event_id:
            raise ValueError("Event missing novel_id or event_id")

        # 获取语义和类型
        semantic = event.get('semantic', 'state_mutation')
        event_type = event.get('type', '')

        # 根据语义设置置信度和优先级
        if semantic in ('dream', 'illusion', 'flashback'):
            base_confidence = 0.4
            base_priority = 'flavor'
        elif semantic in ('dialogue', 'observation'):
            # 对话和观察不产生核心谓词
            return PredicateDelta(
                novel_id=novel_id,
                event_id=event_id,
                projection_version=1,
                event_semantic=semantic,
                to_activate=[],
                to_deactivate=[]
            )
        else:  # state_mutation, intention 等
            base_confidence = 1.0
            base_priority = 'core' if event_type in self._core_event_types() else 'narrative'

        # 根据事件类型生成谓词
        to_activate = []
        to_deactivate = []

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

        elif event_type == 'item_lose':
            actor = event.get('actor')
            item = event.get('item')
            if actor and item:
                # 失效对应的 has_item 谓词
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

        elif event_type == 'realm_upgrade':
            actor = event.get('actor')
            # 新字段：to_major_realm 可能是字符串或枚举值
            to_major_realm = event.get('to_major_realm')
            if actor and to_major_realm:
                # 如果已经是字符串（如"金丹"），直接使用；如果是枚举值，取 value
                if hasattr(to_major_realm, 'value'):
                    realm_str = to_major_realm.value
                else:
                    realm_str = to_major_realm
                normalized_realm = self._normalize_realm(realm_str)
                # 新谓词
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
                logger.debug(f"Will activate realm predicate: ({actor}, realm, {normalized_realm})")
                # 旧境界的失效由 ProjectionStore 中的单值关系处理

        # 其他事件类型可扩展，暂无处理则留空
        # elif event_type == 'hp_changed':
        #     pass

        return PredicateDelta(
            novel_id=novel_id,
            event_id=event_id,
            projection_version=1,
            event_semantic=semantic,
            to_activate=to_activate,
            to_deactivate=to_deactivate
        )

    def _core_event_types(self) -> set:
        """产生核心谓词的事件类型"""
        return {'realm_upgrade', 'item_acquire', 'relationship_change', 'location_enter'}
    
    @staticmethod
    async def rebuild_all_predicates(novel_id: str, pool=None):
        """从事件流全量重建 predicates 表（幂等）"""
        
        from src.writing.event_store import NarrativeEventStore

        pool = pool or get_db_pool()
        if not pool:
            raise RuntimeError("Database pool not available")

        event_store = NarrativeEventStore(pool)
        store = ProjectionStore(pool)

        # 1. 清除该小说的所有 predicates 记录和投影幂等记录
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM predicates WHERE novel_id = $1", novel_id)
            await conn.execute("DELETE FROM projection_applied WHERE novel_id = $1", novel_id)

        # 2. 获取所有事件（按顺序）
        events = await event_store.get_events_since(novel_id, since_event_id=0, limit=1000000)
        current_active: Dict[str, Predicate] = {}  # 内存中的活跃谓词映射
        delta_engine = DeltaEngine()

        for evt_id, evt in events:
            # 将事件转换为字典格式（DeltaEngine 需要的格式）
            event_dict = evt.model_dump()
            event_dict["id"] = evt_id
            event_dict["novel_id"] = novel_id
            # 确保 semantic 字段存在
            if "semantic" not in event_dict:
                event_dict["semantic"] = "state_mutation"

            # 计算 delta
            delta = delta_engine.compute_delta(current_active, event_dict)
            # 应用到数据库（ProjectionStore 会处理幂等和单值关系）
            await store.apply_delta(delta)

            # 手动更新内存中的 current_active（避免重新查询数据库）
            for pred in delta.to_activate:
                current_active[pred.identity_key()] = pred
            for ref in delta.to_deactivate:
                current_active.pop(ref.identity_key, None)

        # 3. 更新 projection_health 表
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
        """
        验证投影一致性：比较当前 predicates 哈希与从事件流重建的哈希。
        
        Args:
            novel_id: 小说 ID
            auto_repair: 若不一致是否自动重建
        
        Returns:
            包含 consistency 状态和哈希值的字典
        """
       
        pool = get_db_pool()
        if not pool:
            raise RuntimeError("Database pool not available")
        
        # 1. 获取当前核心谓词哈希（来自 projection_health 表）
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT core_predicates_hash, last_full_rebuild_at FROM projection_health WHERE novel_id = $1",
                novel_id
            )
        current_hash = row["core_predicates_hash"] if row else None
        
        # 2. 重建并获取新哈希（不保存到数据库，仅计算）
        # 为了不污染现有数据，我们临时重建到内存，但 rebuild_all_predicates 会实际写入。
        # 如果 auto_repair=False，我们不应该实际写入。
        # 因此需要一种“只计算哈希但不写入”的方法。这里暂时先调用 rebuild_all_predicates
        # 但会写入数据库。为了不破坏现有数据，我们在 auto_repair=True 时才重建。
        if auto_repair:
            await DeltaEngine.rebuild_all_predicates(novel_id, pool)
            # 重建后重新获取哈希
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
            # 仅对比，不重建：需要计算从事件流重建后的哈希，但不保存。
            # 实现一个临时重建并计算哈希的函数（不写数据库）
            # 为了简单，我们可以调用 rebuild_all_predicates 但备份当前表，然后恢复。
            # 但这样复杂且低效。替代方案：编写一个纯内存重建函数。
            # 这里我们暂时不实现 auto_repair=False 的内存重建，建议用户直接使用 auto_repair=True 或手动运行 rebuild_all_predicates 后对比。
            return {
                "consistent": None,  # 无法判断，需要手动重建
                "current_hash": current_hash,
                "rebuilt_hash": None,
                "auto_repaired": False,
                "message": "Run with auto_repair=True to perform full rebuild and compare."
            }
