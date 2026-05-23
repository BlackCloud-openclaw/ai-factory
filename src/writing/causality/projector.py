"""DeltaEngine - 纯函数，计算 PredicateDelta"""
import logging
import re
from typing import Dict, List, Optional, Any
from .predicate import Predicate
from .delta import PredicateDelta, PredicateRef

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