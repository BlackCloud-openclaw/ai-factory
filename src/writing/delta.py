"""
StateDelta - 状态增量

由 Planner 生成，是唯一的权威变更来源。
Writer 必须根据 delta 渲染正文，不能自行决定状态变化。
"""
from typing import List, Dict, Any, TYPE_CHECKING
from pydantic import BaseModel, Field

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
)

if TYPE_CHECKING:
    from .world_state import WorldState, CharacterState, ItemState, MapState


class StateDelta(BaseModel):
    """
    状态增量
    
    包含一组类型化事件，表示一次场景/章节引起的状态变更。
    这是系统中最权威的变更单元。
    """
    events: List[NarrativeEvent] = Field(default_factory=list) #移除, min_items=1
    
    def apply_to(self, world_state: 'WorldState') -> 'WorldState':
        """
        应用 delta 到世界状态，返回新状态（纯函数）
        """
        from .world_state import WorldState, CharacterState, ItemState, LocationState
        
        # 深拷贝关键可变字段
        new_state = world_state.model_copy(deep=False)
        
        # 复制 characters 字典，并深拷贝每个 CharacterState
        new_state.characters = {}
        for name, char in world_state.characters.items():
            new_state.characters[name] = char.model_copy(deep=True)
        
        # 复制 items 字典，并深拷贝每个 ItemState
        new_state.items = {}
        for name, item in world_state.items.items():
            new_state.items[name] = item.model_copy(deep=True)
        
        # 复制 relationships 和 global_flags（浅拷贝即可，因为内部是简单类型）
        new_state.relationships = world_state.relationships.copy()
        new_state.global_flags = world_state.global_flags.copy()
        new_state.recent_event_ids = world_state.recent_event_ids.copy()
        
        # 复制 map（深拷贝 LocationState）
        new_state.map = world_state.map.model_copy(deep=True)
        
        new_state.revision += 1
        
        for event in self.events:
            new_state = self._apply_event(new_state, event)
        
        return new_state
    
    def _apply_event(self, state: 'WorldState', event: NarrativeEvent) -> 'WorldState':
        """应用单个事件（内部方法）"""
        from .world_state import WorldState, CharacterState, ItemState, LocationState, Realm
        
        if isinstance(event, RealmUpgradeEvent):
            # 境界突破
            if event.actor in state.characters:
                char = state.characters[event.actor]
                # 更新境界
                char.realm = Realm(event.to_realm)
                char.realm_level = event.to_level
                char.last_active = event.timestamp
            else:
                # 新角色（首次出现）
                state.characters[event.actor] = CharacterState(
                    name=event.actor,
                    realm=Realm(event.to_realm),
                    realm_level=event.to_level,
                    last_active=event.timestamp
                )
        
        elif isinstance(event, ItemAcquireEvent):
            # 获得物品
            if event.actor in state.characters:
                char = state.characters[event.actor]
                # 添加到背包（简化：允许多个相同物品，暂不合并）
                for _ in range(event.quantity):
                    char.inventory.append(event.item)
                char.last_active = event.timestamp
            
            # 添加到全局物品表
            if event.item not in state.items:
                state.items[event.item] = ItemState(
                    name=event.item,
                    owner=event.actor,
                )
            else:
                state.items[event.item].owner = event.actor
        
        elif isinstance(event, ItemLoseEvent):
            # 失去物品
            if event.actor in state.characters:
                char = state.characters[event.actor]
                for _ in range(event.quantity):
                    if event.item in char.inventory:
                        char.inventory.remove(event.item)
                char.last_active = event.timestamp
            
            if event.item in state.items:
                state.items[event.item].owner = None
        
        elif isinstance(event, RelationshipChangeEvent):
            # 关系变化
            key = f"{event.from_char}|{event.to_char}"
            state.relationships[key] = event.new_value
        
        elif isinstance(event, LocationEnterEvent):
            # 进入地点
            if event.actor in state.characters:
                state.characters[event.actor].location = event.location
                state.characters[event.actor].last_active = event.timestamp
            state.map.current = event.location
            
            # 确保地点存在于地图中
            if event.location not in state.map.locations:
                state.map.locations[event.location] = LocationState(name=event.location)
        
        elif isinstance(event, PlotFlagSetEvent):
            # 设置剧情标记
            state.global_flags[event.flag] = event.value
        
        elif isinstance(event, HPChangedEvent):
            # 生命值变化
            if event.actor in state.characters:
                state.characters[event.actor].hp = event.new_hp
                state.characters[event.actor].last_active = event.timestamp
        
        elif isinstance(event, MPChangedEvent):
            # 灵力值变化
            if event.actor in state.characters:
                state.characters[event.actor].mp = event.new_mp
                state.characters[event.actor].last_active = event.timestamp
        
        elif isinstance(event, InventoryAddedEvent):
            if event.actor in state.characters:
                for _ in range(event.quantity):
                    state.characters[event.actor].inventory.append(event.item)
                state.characters[event.actor].last_active = event.timestamp
        
        elif isinstance(event, InventoryRemovedEvent):
            if event.actor in state.characters:
                for _ in range(event.quantity):
                    if event.item in state.characters[event.actor].inventory:
                        state.characters[event.actor].inventory.remove(event.item)
                state.characters[event.actor].last_active = event.timestamp
        
        elif isinstance(event, CombatResultEvent):
            # 复合叙事标记：不修改状态，仅记录摘要（通过 timeline 的摘要实现）
            # 这里留空，未来可扩展
            pass
        
        elif isinstance(event, DialogueEvent):
            # 对话标记：不修改状态
            pass
        
        elif isinstance(event, DiscoveryEvent):
            # 发现标记：可设置对应的 plot_flag
            if event.importance == "critical":
                state.global_flags[f"discovered_{event.discovery}"] = True
        
        elif isinstance(event, NPCIntroduceEvent):
            # NPC 引入：如果角色不存在则创建默认状态
            if event.name not in state.characters:
                from .world_state import CharacterState
                state.characters[event.name] = CharacterState(
                    name=event.name,
                    location=state.map.current,
                    last_active=event.timestamp
                )
                # 添加关系：初次见面，好感度 0
                if event.name not in state.relationships:
                    state.relationships[f"protagonist|{event.name}"] = 0
        
        # 通用：将事件 ID 添加到 recent_event_ids（需要实际存储后获得 ID）
        # 这个由调用方在存储后负责添加
        
        # 更新世界状态根对象的 map 引用（确保返回的新对象中包含所有修改）
        # 注意：上面的操作已直接修改了 state 对象内的可变字典，无需额外操作
        
        return state
    
    def to_prompt_friendly(self) -> Dict[str, Any]:
        """转换为 LLM 友好的格式（用于 prompt）"""
        events_summary = []
        for e in self.events:
            event_dict = e.model_dump(exclude={'event_id', 'timestamp', 'event_version'})
            if 'type' in event_dict and hasattr(event_dict['type'], 'value'):
                event_dict['type'] = event_dict['type'].value
            events_summary.append(event_dict)
        return {"events": events_summary}
    
    def to_dict(self) -> Dict[str, Any]:
        """完整序列化"""
        return {
            "events": [e.model_dump(mode='json') for e in self.events]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateDelta':
        """从字典恢复"""
        from .events import event_from_dict
        
        events = []
        for evt_data in data.get("events", []):
            event_type = evt_data.get("type")
            if isinstance(event_type, Enum):
                event_type = event_type.value
            event = event_from_dict(event_type, evt_data)
            if event:
                events.append(event)
        return cls(events=events)
    
    @classmethod
    def empty(cls) -> 'StateDelta':
        """空 delta（用于无状态变更的场景）"""
        return cls(events=[])