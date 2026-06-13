"""
类型化叙事事件 - 解决 Delta Explosion

所有状态变更必须通过类型化事件表达。
事件是不可变的，一旦创建不应修改。
"""
from typing import Union, List, Optional, Any
from enum import Enum
from datetime import datetime
import uuid
import logging
from pydantic import BaseModel, Field


class EventType(str, Enum):
    """事件类型枚举"""
    # 状态变更事件
    REALM_UPGRADE = "realm_upgrade"
    ITEM_ACQUIRE = "item_acquire"
    ITEM_LOSE = "item_lose"
    RELATIONSHIP_CHANGE = "relationship_change"
    LOCATION_ENTER = "location_enter"
    PLOT_FLAG_SET = "plot_flag_set"
    
    # 原子状态事件
    HP_CHANGED = "hp_changed"
    MP_CHANGED = "mp_changed"
    INVENTORY_ADDED = "inventory_added"
    INVENTORY_REMOVED = "inventory_removed"
    
    # 复合叙事标记（不改变状态，仅用于剧情）
    COMBAT_RESULT = "combat_result"
    DIALOGUE = "dialogue"
    DISCOVERY = "discovery"
    NPC_INTRODUCE = "npc_introduce"

    PERCEPTION_UPDATE = "perception_update"    

class MajorRealm(str, Enum):
    """大境界枚举（不含层级）"""
    QI_REFINING = "炼气"
    FOUNDATION = "筑基"
    GOLDEN_CORE = "金丹"
    NASCENT_SOUL = "元婴"
    DEITY_TRANSFORMATION = "化神"
    VOID_REFINEMENT = "炼虚"
    INTEGRATION = "合体"
    MAHAYANA = "大乘"
    TRIBULATION = "渡劫"

class BaseNarrativeEvent(BaseModel):
    """事件基类"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_version: int = 1
    type: EventType
    timestamp: datetime = Field(default_factory=datetime.now)
    scene_id: Optional[int] = None
    chapter_id: Optional[int] = None

class PerceptionUpdateEvent(BaseNarrativeEvent):
    """认知关系更新事件（由系统自动生成）"""
    type: EventType = EventType.PERCEPTION_UPDATE
    observer: str          # 观察者
    target: str            # 被观察的角色
    new_value: int         # 新的认知值 (-100..100)
    confidence_delta: float = 0.0   # 确信度变化（增量，最终值会被钳位）
    reason: str = ""       # 更新原因（观察、对话、推理）

# ========== 状态变更事件 ==========

class RealmUpgradeEvent(BaseNarrativeEvent):
    type: EventType = EventType.REALM_UPGRADE
    actor: str
    from_realm: Optional[MajorRealm] = None   # 可选，因为可能是首次出现
    from_level: Optional[int] = None
    to_major_realm: MajorRealm                # 目标大境界
    to_minor_stage: int                       # 1-9 小层级
    breakthrough_method: str = "normal"


class ItemAcquireEvent(BaseNarrativeEvent):
    """获得物品事件"""
    type: EventType = EventType.ITEM_ACQUIRE
    actor: str
    item: str
    source: Optional[str] = None  # 来源：宝箱、击败敌人、赠送等
    quantity: int = 1


class ItemLoseEvent(BaseNarrativeEvent):
    """失去物品事件"""
    type: EventType = EventType.ITEM_LOSE
    actor: str
    item: str
    reason: str = ""
    quantity: int = 1


class RelationshipChangeEvent(BaseNarrativeEvent):
    """关系变化事件"""
    type: EventType = EventType.RELATIONSHIP_CHANGE
    from_char: str
    to_char: str
    delta: int  # -100 到 100
    new_value: int
    reason: str = ""


class LocationEnterEvent(BaseNarrativeEvent):
    """进入地点事件"""
    type: EventType = EventType.LOCATION_ENTER
    actor: str
    location: str
    first_time: bool = False


class PlotFlagSetEvent(BaseNarrativeEvent):
    """剧情标记设置事件"""
    type: EventType = EventType.PLOT_FLAG_SET
    flag: str
    value: Any = True


# ========== 原子状态事件 ==========

class HPChangedEvent(BaseNarrativeEvent):
    """生命值变化事件"""
    type: EventType = EventType.HP_CHANGED
    actor: str
    delta: int
    new_hp: int


class MPChangedEvent(BaseNarrativeEvent):
    """灵力值变化事件"""
    type: EventType = EventType.MP_CHANGED
    actor: str
    delta: int
    new_mp: int


class InventoryAddedEvent(BaseNarrativeEvent):
    """背包添加物品事件"""
    type: EventType = EventType.INVENTORY_ADDED
    actor: str
    item: str
    quantity: int = 1


class InventoryRemovedEvent(BaseNarrativeEvent):
    """背包移除物品事件"""
    type: EventType = EventType.INVENTORY_REMOVED
    actor: str
    item: str
    quantity: int = 1


# ========== 复合叙事标记 ==========

class CombatResultEvent(BaseNarrativeEvent):
    """战斗结果事件（复合标记，不改变状态）"""
    type: EventType = EventType.COMBAT_RESULT
    winner: str
    loser: str
    result: str  # 胜利/失败/平局/逃脱
    casualties: List[str] = Field(default_factory=list)
    loot: List[str] = Field(default_factory=list)  # 战利品


class DialogueEvent(BaseNarrativeEvent):
    """对话事件（复合标记）"""
    type: EventType = EventType.DIALOGUE
    speaker: str
    listener: str
    summary: str  # 对话摘要
    key_revelation: Optional[str] = None  # 关键信息


class DiscoveryEvent(BaseNarrativeEvent):
    """发现事件（复合标记）"""
    type: EventType = EventType.DISCOVERY
    discoverer: str
    discovery: str  # 发现了什么
    importance: str = "normal"  # low, normal, high, critical


class NPCIntroduceEvent(BaseNarrativeEvent):
    """NPC 引入事件"""
    type: EventType = EventType.NPC_INTRODUCE
    name: str
    role: str  # 身份：长老、同门、敌人等
    realm: Optional[str] = None
    first_impression: str = ""


# 类型联合
NarrativeEvent = Union[
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
    PerceptionUpdateEvent,
]


# ========== 辅助函数 ==========

def event_to_dict(event: NarrativeEvent) -> dict:
    """将事件转换为字典（用于存储）"""
    return event.model_dump(mode='json')


def event_from_dict(event_type: str, data: dict) -> Optional[NarrativeEvent]:
    """从字典恢复事件，带容错处理"""
    event_map = {
        "realm_upgrade": RealmUpgradeEvent,
        "item_acquire": ItemAcquireEvent,
        "item_lose": ItemLoseEvent,
        "relationship_change": RelationshipChangeEvent,
        "location_enter": LocationEnterEvent,
        "plot_flag_set": PlotFlagSetEvent,
        "hp_changed": HPChangedEvent,
        "mp_changed": MPChangedEvent,
        "inventory_added": InventoryAddedEvent,
        "inventory_removed": InventoryRemovedEvent,
        "combat_result": CombatResultEvent,
        "dialogue": DialogueEvent,
        "discovery": DiscoveryEvent,
        "npc_introduce": NPCIntroduceEvent,
        "item_discovery": DiscoveryEvent,
        "perception_update": PerceptionUpdateEvent,
    }
    
    cls = event_map.get(event_type)
    if cls is None:
        # 未知事件类型，记录警告并返回 None
        logger = logging.getLogger("writing.events")
        logger.warning(f"Unknown event type: {event_type}, data: {data}")
        return None
    
    try:
        return cls.model_validate(data)
    except Exception as e:
        # 解析失败，记录警告并返回 None
        logger = logging.getLogger("writing.events")
        logger.warning(f"Failed to parse event {event_type}: {e}")
        logger.debug(f"Event data: {data}")
        return None