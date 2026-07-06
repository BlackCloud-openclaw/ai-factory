# src/writing/delta.py
"""
StateDelta - 状态增量（写时复制优化版）

Phase 4C 迁移：
- 所有写操作使用 _get_character() + _set_character()
- characters 字典键统一使用 character_id
- 保留 name-key 双写兼容（可逐步移除）
"""

import logging
from typing import List, Dict, Any, TYPE_CHECKING
from pydantic import BaseModel, Field
from copy import deepcopy

from .events import (
    NarrativeEvent,
    RealmUpgradeEvent,
    ItemAcquireEvent,
    ItemLoseEvent,
    RelationshipChangeEvent,
    LocationEnterEvent,
    PlotFlagSetEvent,
    HPChangedEvent,
    MPChangedEvent,
    InventoryAddedEvent,
    InventoryRemovedEvent,
    CombatResultEvent,
    DialogueEvent,
    DiscoveryEvent,
    NPCIntroduceEvent,
    event_from_dict,
    MajorRealm,
    PerceptionUpdateEvent,
)
from src.domain.identity import get_character_id_by_name, get_character_name, get_main_character_id

if TYPE_CHECKING:
    from .world_state import WorldState, CharacterState, ItemState, MapState, LocationState

logger = logging.getLogger(__name__)


class StateDelta(BaseModel):
    events: List[NarrativeEvent] = Field(default_factory=list)

    # =========================================================
    # Phase 4C 辅助方法
    # =========================================================
    def _get_character(self, state: 'WorldState', actor: str):
        """通过 actor（名称或 ID）获取角色，使用 get_character API"""
        return state.get_character(actor)

    def _set_character(self, state: 'WorldState', actor: str, char: 'CharacterState'):
        """
        将角色存入 state.characters，key 优先使用 char.id
        同时保留 name-key 双写（兼容期，后续可移除）
        """
        # 1. 确保角色有 id
        if not char.id:
            # 尝试从 actor 或 name 推断 ID
            char_id = get_character_id_by_name(actor)
            if not char_id and hasattr(char, 'name') and char.name:
                char_id = get_character_id_by_name(char.name)
            if not char_id:
                # 如果 actor 本身是 ID（如 "protagonist"），直接使用
                char_id = actor
            char.id = char_id

        # 2. 使用 id 作为 key
        key = char.id
        # 如果 key 为空，fallback 到 actor
        if not key:
            key = actor

        # 3. 写入 state.characters（使用 ID-key）
        state.characters[key] = char

        # 4. 兼容期：同时保留 name-key（便于过渡，后续可移除）
        if actor != key:
            state.characters[actor] = char

        return state

    # =========================================================
    # 原有 apply_to 方法（修改后）
    # =========================================================
    def apply_to(self, world_state: 'WorldState') -> 'WorldState':
        """
        写时复制：只复制被修改的部分
        """
        from .world_state import WorldState, CharacterState, ItemState, LocationState, Realm

        # 浅拷贝顶层结构
        new_state = world_state.model_copy(deep=False)

        # 初始化可变字段为浅拷贝（后续修改时会按需深拷贝）
        new_state.characters = dict(world_state.characters)  # 浅拷贝
        new_state.items = dict(world_state.items)
        new_state.relationships = world_state.relationships.copy()
        new_state.global_flags = world_state.global_flags.copy()
        new_state.recent_event_ids = world_state.recent_event_ids.copy()
        new_state.map = world_state.map.model_copy(deep=False)
        new_state.map.locations = dict(world_state.map.locations)  # 浅拷贝

        new_state.revision += 1

        for event in self.events:
            new_state = self._apply_event_cow(new_state, event)

        return new_state

    def _apply_event_cow(self, state: 'WorldState', event: NarrativeEvent) -> 'WorldState':
        """写时复制版本的事件应用，使用 _get_character 和 _set_character"""
        from .world_state import WorldState, CharacterState, ItemState, LocationState, Realm

        # 按事件类型处理
        if isinstance(event, RealmUpgradeEvent):
            # 将 MajorRealm 枚举值转换为 WorldState 中的 Realm 枚举
            major_realm_map = {
                MajorRealm.QI_REFINING: Realm.REFINING_QI,
                MajorRealm.FOUNDATION: Realm.FOUNDATION,
                MajorRealm.GOLDEN_CORE: Realm.GOLDEN_CORE,
                MajorRealm.NASCENT_SOUL: Realm.NASCENT_SOUL,
                MajorRealm.DEITY_TRANSFORMATION: Realm.DEITY_TRANSFORMATION,
                MajorRealm.VOID_REFINEMENT: Realm.VOID_REFINEMENT,
                MajorRealm.INTEGRATION: Realm.INTEGRATION,
                MajorRealm.MAHAYANA: Realm.MAHAYANA,
                MajorRealm.TRIBULATION: Realm.TRIBULATION,
            }
            target_realm_enum = major_realm_map.get(event.to_major_realm, Realm.REFINING_QI)

            char = self._get_character(state, event.actor)
            if char is not None:
                new_char = char.model_copy(deep=True)
                new_char.realm = target_realm_enum
                new_char.realm_level = event.to_minor_stage
                new_char.last_active = event.timestamp
                self._set_character(state, event.actor, new_char)
            else:
                # 创建新角色
                new_char = CharacterState(
                    name=event.actor,
                    realm=target_realm_enum,
                    realm_level=event.to_minor_stage,
                    last_active=event.timestamp
                )
                self._set_character(state, event.actor, new_char)

        elif isinstance(event, ItemAcquireEvent):
            # 确保角色存在
            char = self._get_character(state, event.actor)
            if char is None:
                char = CharacterState(name=event.actor, last_active=event.timestamp)
                self._set_character(state, event.actor, char)

            # 重新获取以确保有 id
            char = self._get_character(state, event.actor)
            if char is not None:
                new_char = char.model_copy(deep=True)
                for _ in range(event.quantity):
                    new_char.inventory.append(event.item)
                new_char.last_active = event.timestamp
                self._set_character(state, event.actor, new_char)

                # 更新 items 表
                new_items = dict(state.items)
                if event.item not in new_items:
                    new_items[event.item] = ItemState(name=event.item, owner=event.actor)
                else:
                    new_items[event.item].owner = event.actor
                state.items = new_items

        elif isinstance(event, ItemLoseEvent):
            char = self._get_character(state, event.actor)
            if char is not None:
                new_char = char.model_copy(deep=True)
                for _ in range(event.quantity):
                    if event.item in new_char.inventory:
                        new_char.inventory.remove(event.item)
                    else:
                        logger.warning(f"ItemLoseEvent: item '{event.item}' not in inventory of '{event.actor}'")
                new_char.last_active = event.timestamp
                self._set_character(state, event.actor, new_char)

                if event.item in state.items:
                    new_items = dict(state.items)
                    new_items[event.item].owner = None
                    state.items = new_items

        elif isinstance(event, RelationshipChangeEvent):
            key = f"{event.from_char}|{event.to_char}"
            new_rels = dict(state.relationships)
            new_rels[key] = event.new_value
            state.relationships = new_rels

        elif isinstance(event, LocationEnterEvent):
            char = self._get_character(state, event.actor)
            if char is not None:
                new_char = char.model_copy(deep=True)
                new_char.location = event.location
                new_char.last_active = event.timestamp
                self._set_character(state, event.actor, new_char)

            # 更新地图当前地点
            state.map.current = event.location
            # 确保地点存在
            if event.location not in state.map.locations:
                new_locs = dict(state.map.locations)
                new_locs[event.location] = LocationState(name=event.location)
                state.map.locations = new_locs

        elif isinstance(event, PlotFlagSetEvent):
            new_flags = dict(state.global_flags)
            new_flags[event.flag] = event.value
            state.global_flags = new_flags

        elif isinstance(event, HPChangedEvent):
            char = self._get_character(state, event.actor)
            if char is not None:
                new_char = char.model_copy(deep=True)
                new_char.hp = max(0, event.new_hp)   # 钳位
                new_char.last_active = event.timestamp
                self._set_character(state, event.actor, new_char)

        elif isinstance(event, MPChangedEvent):
            char = self._get_character(state, event.actor)
            if char is not None:
                new_char = char.model_copy(deep=True)
                new_char.mp = max(0, event.new_mp)   # 钳位
                new_char.last_active = event.timestamp
                self._set_character(state, event.actor, new_char)

        elif isinstance(event, InventoryAddedEvent):
            char = self._get_character(state, event.actor)
            if char is not None:
                new_char = char.model_copy(deep=True)
                for _ in range(event.quantity):
                    new_char.inventory.append(event.item)
                new_char.last_active = event.timestamp
                self._set_character(state, event.actor, new_char)

        elif isinstance(event, InventoryRemovedEvent):
            char = self._get_character(state, event.actor)
            if char is not None:
                new_char = char.model_copy(deep=True)
                for _ in range(event.quantity):
                    if event.item in new_char.inventory:
                        new_char.inventory.remove(event.item)
                new_char.last_active = event.timestamp
                self._set_character(state, event.actor, new_char)

        elif isinstance(event, PerceptionUpdateEvent):
            char = self._get_character(state, event.observer)
            if char is not None:
                new_char = char.model_copy(deep=True)
                current = new_char.perceived_relationships.get(event.target, {"value": 0, "confidence": 0.0})
                new_value = event.new_value
                new_confidence = min(1.0, max(0.0, current.get("confidence", 0.0) + event.confidence_delta))
                new_char.perceived_relationships[event.target] = {
                    "value": new_value,
                    "confidence": new_confidence
                }
                new_char.last_active = event.timestamp
                self._set_character(state, event.observer, new_char)

        elif isinstance(event, CombatResultEvent):
            # 复合事件，不修改状态
            pass

        elif isinstance(event, DialogueEvent):
            pass

        elif isinstance(event, DiscoveryEvent):
            if event.importance == "critical":
                new_flags = dict(state.global_flags)
                new_flags[f"discovered_{event.discovery}"] = True
                state.global_flags = new_flags

        elif isinstance(event, NPCIntroduceEvent):
            char = self._get_character(state, event.name)
            if char is None:
                new_char = CharacterState(
                    name=event.name,
                    location=state.map.current,
                    last_active=event.timestamp
                )
                self._set_character(state, event.name, new_char)
                # 添加关系
                rel_key = f"protagonist|{event.name}"
                new_rels = dict(state.relationships)
                if rel_key not in new_rels:
                    new_rels[rel_key] = 0
                state.relationships = new_rels

        # 将事件 ID 添加到 recent_event_ids（由调用方处理）
        return state

    # =========================================================
    # 原有类方法（保持不变）
    # =========================================================
    @classmethod
    def from_dict(cls, data: Dict[str, Any], strict: bool = False) -> "StateDelta":
        """从字典创建 StateDelta 实例"""
        events_data = data.get("events", [])
        events = []
        for evt_dict in events_data:
            event_type = evt_dict.get("type")
            if not event_type:
                msg = f"Event dictionary missing 'type' field: {evt_dict}"
                if strict:
                    raise ValueError(msg)
                logging.getLogger(__name__).warning(msg)
                continue
            evt = event_from_dict(event_type, evt_dict)
            if evt is None:
                msg = f"Failed to convert event type '{event_type}' from dict: {evt_dict}"
                if strict:
                    raise ValueError(msg)
                logging.getLogger(__name__).warning(msg)
                continue
            events.append(evt)
        return cls(events=events)