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


# Phase 16.0: LLM 自由 type → 已知 type 的宽松映射
_FREEFORM_TYPE_MAP = {
    "conflict": "plot_flag_set",
    "conflict_escalation": "plot_flag_set",
    "encounter": "plot_flag_set",
    "meeting": "plot_flag_set",
    "activation": "plot_flag_set",
    "trigger": "plot_flag_set",
    "revelation": "plot_flag_set",
    "discovery_reveal": "plot_flag_set",
    "clue_reveal": "plot_flag_set",
    "clue": "plot_flag_set",
    "power_shift": "plot_flag_set",
    "antagonist_appearance": "plot_flag_set",
    "foreshadowing": "plot_flag_set",
    "danger": "plot_flag_set",
    "environmental_trap": "plot_flag_set",
    "biological_trap": "plot_flag_set",
    "situation_escalation": "plot_flag_set",
    "plot_twist": "plot_flag_set",
    "event_trigger": "plot_flag_set",
    "location_change": "location_enter",
    "arrival": "location_enter",
    "escape": "location_enter",
    "departure": "location_enter",
    "enter": "location_enter",
    "breakthrough": "realm_upgrade",
    "realm_change": "realm_upgrade",
    "realm_advance": "realm_upgrade",
    "cultivation_breakthrough": "realm_upgrade",
    "inventory_acquire": "item_acquire",
    "acquisition": "item_acquire",
    "inventory_added": "item_acquire",
    "relation_change": "relationship_change",
    "relationship_shift": "relationship_change",
    "knowledge_discovery": "discovery",
    "knowledge_gain": "discovery",
    "information_retrieval": "discovery",
    "insight": "discovery",
    "comprehension": "discovery",
}


def event_from_dict(event_type: str, data: dict) -> Optional[NarrativeEvent]:
    """从字典恢复事件，带容错处理（Phase 16.0 宽松版）"""
    _log = logging.getLogger("writing.events")

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

    # 1. 自由 type 归一化
    normalized_type = _FREEFORM_TYPE_MAP.get(event_type, event_type)

    cls = event_map.get(normalized_type)

    # 2. 未知 type → fallback 到 plot_flag_set
    if cls is None:
        _log.info(
            "[16.0] event_from_dict fallback: unknown type '%s' → plot_flag_set",
            event_type,
        )
        flag_name = str(event_type or "unknown")[:30]
        return PlotFlagSetEvent(flag=flag_name, value=True)

    # 3. 补齐 plot_flag_set 的 flag 字段（LLM 常用 name / target）
    if normalized_type == "plot_flag_set":
        if "flag" not in data:
            for k in ("name", "target", "description", "event"):
                if isinstance(data.get(k), str) and data[k]:
                    data = {**data, "flag": data[k][:40]}
                    break
            else:
                data = {**data, "flag": f"auto_{abs(hash(str(data))) % 10000}"}

    # 4. 补齐 discovery 的 discoverer / discovery 字段
    if normalized_type == "discovery":
        if "discoverer" not in data:
            data = {**data, "discoverer": data.get("actor", "林逸")}
        if "discovery" not in data:
            for k in ("target", "name", "description", "content"):
                if isinstance(data.get(k), str) and data[k]:
                    data = {**data, "discovery": data[k][:80]}
                    break
            else:
                data = {**data, "discovery": f"发现_{abs(hash(str(data))) % 10000}"}

    # 5. 补齐 location_enter 的 location / actor
    if normalized_type == "location_enter":
        if "actor" not in data:
            data = {**data, "actor": data.get("discoverer", "林逸")}
        if "location" not in data:
            data = {**data, "location": data.get("target", "未知地点")[:40]}

    # 6. 补齐 relationship_change
    if normalized_type == "relationship_change":
        if "from_char" not in data:
            data = {**data, "from_char": data.get("actor", "林逸")}
        if "to_char" not in data:
            data = {**data, "to_char": data.get("target", "未知")[:20]}
        if "delta" not in data:
            data = {**data, "delta": -10}
        if "new_value" not in data:
            data = {**data, "new_value": 0}

    # 7. 补齐 realm_upgrade
    if normalized_type == "realm_upgrade":
        if "actor" not in data:
            data = {**data, "actor": data.get("discoverer", "林逸")}
        if "to_major_realm" not in data:
            data = {**data, "to_major_realm": "金丹"}
        if "to_minor_stage" not in data:
            data = {**data, "to_minor_stage": 1}

    # 8. 补齐 item_acquire
    if normalized_type == "item_acquire":
        if "actor" not in data:
            data = {**data, "actor": "林逸"}
        if "item" not in data:
            data = {**data, "item": data.get("target", "未知物品")[:40]}

    try:
        return cls.model_validate(data)
    except Exception as e:
        _log.info(
            "[16.0] event_from_dict fallback: parse failed for '%s' (%s) → plot_flag_set",
            event_type, type(e).__name__,
        )
        return PlotFlagSetEvent(
            flag=f"fallback_{event_type or 'unknown'}_{abs(hash(str(data))) % 10000}",
            value=True,
        )