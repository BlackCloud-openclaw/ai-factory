# src/writing/causality/initializer.py

import logging
import re
from typing import List
from src.db import get_db_pool
from .predicate import Predicate
from .delta import PredicateDelta, PredicateRef
from .projection_store import ProjectionStore
from src.writing.world_state import WorldState
from src.domain.identity import get_main_character_id, get_character_name

logger = logging.getLogger(__name__)

def normalize_realm(realm_str: str) -> str:
    """将境界字符串规范化（移除层级后缀）"""
    if not realm_str:
        return realm_str
    # 移除中文数字 + "层"、"期"、"重"、"级" 等后缀
    cleaned = re.sub(r'[一二三四五六七八九零\d]+(?:层|期|重|级)?$', '', realm_str)
    if not cleaned:
        return realm_str
    return cleaned.strip()

async def ensure_core_predicates(novel_id: str, world_state: WorldState) -> None:
    """
    确保世界状态中的核心关系已经投影到 predicates 表。
    使用配置中的主角 ID 和名称。
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

    # ... 辅助函数 normalize_realm 保持不变 ...

    # 构建待激活的谓词列表
    to_activate: List[Predicate] = []

    # 获取主角信息
    protagonist_id = get_main_character_id()
    protagonist_name = get_character_name(protagonist_id)

    for char_name, char_state in world_state.characters.items():
        # 判断是否为当前主角（通过 ID 或名称）
        is_protagonist = (
            hasattr(char_state, 'id') and char_state.id == protagonist_id
        ) or char_name == protagonist_name

        # realm 谓词
        if char_state.realm:
            full_realm = char_state.full_realm()
            base_realm = normalize_realm(full_realm)
            if not base_realm:
                base_realm = char_state.realm.value
            priority = 'core' if is_protagonist else 'narrative'
            to_activate.append(Predicate(
                subject=char_name,
                relation="realm",
                object=base_realm,
                confidence=1.0,
                priority=priority,
                source_event_type="system_init",
                source_event_semantic="state_mutation"
            ))

        # is_alive 谓词（所有角色默认为存活）
        to_activate.append(Predicate(
            subject=char_name,
            relation="is_alive",
            object=True,
            confidence=1.0,
            priority="core" if is_protagonist else "narrative",
            source_event_type="system_init",
            source_event_semantic="state_mutation"
        ))

        # location 谓词
        if char_state.location:
            to_activate.append(Predicate(
                subject=char_name,
                relation="location",
                object=char_state.location,
                confidence=1.0,
                priority="core" if is_protagonist else "narrative",
                source_event_type="system_init",
                source_event_semantic="state_mutation"
            ))

    if not to_activate:
        logger.warning(f"No core predicates to activate for novel {novel_id}")
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