"""
StateDelta - 状态增量（写时复制优化版）
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

if TYPE_CHECKING:
    from .world_state import WorldState, CharacterState, ItemState, MapState, LocationState

logger = logging.getLogger(__name__)


class StateDelta(BaseModel):
    events: List[NarrativeEvent] = Field(default_factory=list)
    
    def apply_to(self, world_state: 'WorldState') -> 'WorldState':
        """
        写时复制：只复制被修改的部分
        """
        from .world_state import WorldState, CharacterState, ItemState, LocationState, Realm
        
        # 浅拷贝顶层结构（字典只复制引用，但我们需要在修改时复制条目）
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
        """写时复制版本的事件应用，返回新状态"""
        from .world_state import WorldState, CharacterState, ItemState, LocationState, Realm
        
        # 按事件类型处理，每个分支都确保只复制被修改的对象
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
                # 后续境界根据实际 Realm 枚举补充
            }
            target_realm_enum = major_realm_map.get(event.to_major_realm, Realm.REFINING_QI)
            
            if event.actor in state.characters:
                old_char = state.characters[event.actor]
                # 复制该角色
                new_char = old_char.model_copy(deep=True)
                new_char.realm = target_realm_enum
                new_char.realm_level = event.to_minor_stage
                new_char.last_active = event.timestamp
                # 替换字典中的条目
                new_chars = dict(state.characters)
                new_chars[event.actor] = new_char
                state.characters = new_chars
            else:
                # 新角色
                new_char = CharacterState(
                    name=event.actor,
                    realm=target_realm_enum,          # 使用已转换的枚举
                    realm_level=event.to_minor_stage,  # 使用新字段
                    last_active=event.timestamp
                )
                new_chars = dict(state.characters)
                new_chars[event.actor] = new_char
                state.characters = new_chars
        
        elif isinstance(event, ItemAcquireEvent):
            # 确保角色存在
            if event.actor not in state.characters:
                logger.warning(f"ItemAcquireEvent: actor '{event.actor}' not found, creating")
                new_char = CharacterState(name=event.actor, last_active=event.timestamp)
                new_chars = dict(state.characters)
                new_chars[event.actor] = new_char
                state.characters = new_chars
            
            old_char = state.characters[event.actor]
            new_char = old_char.model_copy(deep=True)
            for _ in range(event.quantity):
                new_char.inventory.append(event.item)
            new_char.last_active = event.timestamp
            new_chars = dict(state.characters)
            new_chars[event.actor] = new_char
            state.characters = new_chars
            
            # 更新 items 表
            new_items = dict(state.items)
            if event.item not in new_items:
                new_items[event.item] = ItemState(name=event.item, owner=event.actor)
            else:
                new_items[event.item].owner = event.actor
            state.items = new_items
        
        elif isinstance(event, ItemLoseEvent):
            if event.actor in state.characters:
                old_char = state.characters[event.actor]
                new_char = old_char.model_copy(deep=True)
                for _ in range(event.quantity):
                    if event.item in new_char.inventory:
                        new_char.inventory.remove(event.item)
                    else:
                        logger.warning(f"ItemLoseEvent: item '{event.item}' not in inventory of '{event.actor}'")
                new_char.last_active = event.timestamp
                new_chars = dict(state.characters)
                new_chars[event.actor] = new_char
                state.characters = new_chars
            
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
            if event.actor in state.characters:
                old_char = state.characters[event.actor]
                new_char = old_char.model_copy(deep=True)
                new_char.location = event.location
                new_char.last_active = event.timestamp
                new_chars = dict(state.characters)
                new_chars[event.actor] = new_char
                state.characters = new_chars
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
            if event.actor in state.characters:
                old_char = state.characters[event.actor]
                new_char = old_char.model_copy(deep=True)
                new_char.hp = max(0, event.new_hp)   # 钳位
                new_char.last_active = event.timestamp
                new_chars = dict(state.characters)
                new_chars[event.actor] = new_char
                state.characters = new_chars
        
        elif isinstance(event, MPChangedEvent):
            if event.actor in state.characters:
                old_char = state.characters[event.actor]
                new_char = old_char.model_copy(deep=True)
                new_char.mp = max(0, event.new_mp)   # 钳位
                new_char.last_active = event.timestamp
                new_chars = dict(state.characters)
                new_chars[event.actor] = new_char
                state.characters = new_chars
        
        elif isinstance(event, InventoryAddedEvent):
            if event.actor in state.characters:
                old_char = state.characters[event.actor]
                new_char = old_char.model_copy(deep=True)
                for _ in range(event.quantity):
                    new_char.inventory.append(event.item)
                new_char.last_active = event.timestamp
                new_chars = dict(state.characters)
                new_chars[event.actor] = new_char
                state.characters = new_chars
        
        elif isinstance(event, InventoryRemovedEvent):
            if event.actor in state.characters:
                old_char = state.characters[event.actor]
                new_char = old_char.model_copy(deep=True)
                for _ in range(event.quantity):
                    if event.item in new_char.inventory:
                        new_char.inventory.remove(event.item)
                new_char.last_active = event.timestamp
                new_chars = dict(state.characters)
                new_chars[event.actor] = new_char
                state.characters = new_chars

        elif isinstance(event, PerceptionUpdateEvent):
            if event.observer in state.characters:
                old_char = state.characters[event.observer]
                new_char = old_char.model_copy(deep=True)
                # 获取当前认知
                current = new_char.perceived_relationships.get(event.target, {"value": 0, "confidence": 0.0})
                # 更新值
                new_value = event.new_value
                # 更新确信度（增量并钳位）
                new_confidence = min(1.0, max(0.0, current.get("confidence", 0.0) + event.confidence_delta))
                new_char.perceived_relationships[event.target] = {
                    "value": new_value,
                    "confidence": new_confidence
                }
                new_char.last_active = event.timestamp
                new_chars = dict(state.characters)
                new_chars[event.observer] = new_char
                state.characters = new_chars

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
            if event.name not in state.characters:
                new_char = CharacterState(
                    name=event.name,
                    location=state.map.current,
                    last_active=event.timestamp
                )
                new_chars = dict(state.characters)
                new_chars[event.name] = new_char
                state.characters = new_chars
                # 添加关系
                rel_key = f"protagonist|{event.name}"
                new_rels = dict(state.relationships)
                if rel_key not in new_rels:
                    new_rels[rel_key] = 0
                state.relationships = new_rels
        
        # 将事件 ID 添加到 recent_event_ids（由调用方处理）
        return state
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], strict: bool = False) -> "StateDelta":
        """从字典创建 StateDelta 实例。
        
        Args:
            data: 包含 "events" 键的字典，值为事件字典列表。
            strict: 若为 True，遇到无效事件则抛出异常；否则跳过并记录日志。
        """

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