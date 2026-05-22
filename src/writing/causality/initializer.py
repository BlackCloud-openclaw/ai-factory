# src/writing/causality/initializer.py

import logging
import re
from typing import List
from src.db import get_db_pool
from .predicate import Predicate
from .delta import PredicateDelta, PredicateRef
from .projection_store import ProjectionStore
from src.writing.world_state import WorldState

logger = logging.getLogger(__name__)


async def ensure_core_predicates(novel_id: str, world_state: WorldState) -> None:
    """
    确保世界状态中的核心关系（realm, is_alive, location）已经投影到 predicates 表。
    幂等：如果已存在则跳过，否则插入。
    使用大境界（不含层级）作为 realm 的 object，与 DeltaEngine 保持一致。
    """
    pool = get_db_pool()
    if not pool:
        logger.error("Database pool not available, cannot ensure core predicates")
        return

    # 检查是否已经存在该小说的任何核心谓词
    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT 1 FROM predicates WHERE novel_id = $1 AND relation IN ('realm', 'is_alive', 'location') LIMIT 1",
            novel_id
        )
        if exists:
            logger.info(f"Core predicates already exist for novel {novel_id}, skipping initialization")
            return

    # 辅助函数：将境界规范化为大境界
    def normalize_realm(realm_str: str) -> str:
        if not realm_str:
            return realm_str
        # 移除数字（中文或阿拉伯）和"层"、"期"等后缀
        cleaned = re.sub(r'[一二三四五六七八九零\d]+(?:层|期|重|级)?$', '', realm_str)
        if not cleaned:
            return realm_str
        return cleaned.strip()

    # 构建待激活的谓词列表
    to_activate: List[Predicate] = []
    for char_name, char_state in world_state.characters.items():
        # realm 谓词（使用大境界，不含层级）
        if char_state.realm:
            # 获取完整境界描述，再规范化为大境界
            full_realm = char_state.full_realm()  # 例如 "炼气一层"
            base_realm = normalize_realm(full_realm)
            # 如果没有提取到大境界，使用 realm 枚举值
            if not base_realm:
                base_realm = char_state.realm.value
            logger.debug(f"Character {char_name}: full_realm={full_realm}, base_realm={base_realm}")
            to_activate.append(Predicate(
                subject=char_name,
                relation="realm",
                object=base_realm,
                confidence=1.0,
                priority="core",
                source_event_type="system_init",
                source_event_semantic="state_mutation"
            ))
        # is_alive 谓词（所有角色默认为存活，除非特殊标记）
        to_activate.append(Predicate(
            subject=char_name,
            relation="is_alive",
            object=True,
            confidence=1.0,
            priority="core",
            source_event_type="system_init",
            source_event_semantic="state_mutation"
        ))
        # location 谓词（如果角色有位置）
        if char_state.location:
            to_activate.append(Predicate(
                subject=char_name,
                relation="location",
                object=char_state.location,
                confidence=1.0,
                priority="core",
                source_event_type="system_init",
                source_event_semantic="state_mutation"
            ))

    if not to_activate:
        logger.warning(f"No core predicates to activate for novel {novel_id} (no characters in world_state?)")
        return

    # 构造虚拟 Delta（使用 event_id = 0 表示初始化）
    delta = PredicateDelta(
        novel_id=novel_id,
        event_id=0,
        projection_version=1,
        event_semantic="state_mutation",
        to_activate=to_activate,
        to_deactivate=[]
    )

    store = ProjectionStore(pool)
    success = await store.apply_delta(delta)
    if success:
        logger.info(f"✅ Initialized {len(to_activate)} core predicates for novel {novel_id}")
        logger.debug(f"Predicates: {[(p.subject, p.relation, p.object) for p in to_activate]}")
    else:
        logger.error(f"❌ Failed to initialize core predicates for novel {novel_id}")