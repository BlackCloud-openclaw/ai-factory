"""
规范化的世界状态模型 - 唯一真状态
"""
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
from .delta import StateDelta
from src.common.canonical import canonical_hash
from pydantic import BaseModel, Field, field_validator, ValidationInfo

class Realm(str, Enum):
    """境界枚举"""
    MORTAL = "凡人"
    REFINING_QI = "炼气"
    FOUNDATION = "筑基"
    GOLDEN_CORE = "金丹"
    NASCENT_SOUL = "元婴"
    DEITY_TRANSFORMATION = "化神"
    VOID_REFINEMENT = "炼虚"
    INTEGRATION = "合体"
    MAHAYANA = "大乘"
    TRIBULATION = "渡劫"


class CharacterState(BaseModel):
    """角色状态 - 包含客观属性、认知关系和身份"""
    name: str
    realm: Realm = Realm.REFINING_QI
    realm_level: int = Field(1, ge=1, le=9)
    hp: int = Field(100, ge=0)
    mp: int = Field(100, ge=0)
    inventory: List[str] = Field(default_factory=list)
    relationships: Dict[str, int] = Field(default_factory=dict)  # 客观关系
    perceived_relationships: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="认知关系，格式: {target: {'value': int, 'confidence': float}}"
    )
    location: str = ""
    flags: Dict[str, Any] = Field(default_factory=dict)
    last_active: datetime = Field(default_factory=datetime.now)
    
    # ========== 认知身份（新增）==========
    beliefs: List[str] = Field(
        default_factory=list,
        description="核心信念，如['强者为尊', '丹药不可靠', '不背叛朋友']"
    )
    attachments: List[str] = Field(
        default_factory=list,
        description="依恋对象/物品，如['二叔', '神秘玉佩', '青云宗']"
    )
    self_image: str = Field(
        default="",
        description="自我认知，如'天弃之子'、'复仇者'、'守护者'"
    )
    moral_boundaries: List[str] = Field(
        default_factory=list,
        description="道德底线，如['不杀无辜', '不背叛宗门', '不食言']"
    )
    # ====================================

    @field_validator('realm_level')
    def level_within_bounds(cls, v, info: ValidationInfo):
        values = info.data
        if 'realm' in values and values['realm'] != Realm.MORTAL:
            if not (1 <= v <= 9):
                raise ValueError(f'境界层级必须在1-9之间，当前: {v}')
        return v    
    
    def full_realm(self) -> str:
        """完整境界描述，如'炼气三层'"""
        if self.realm == Realm.MORTAL:
            return "凡人"
        if self.realm_level == 0:
            return self.realm.value
        level_chars = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]
        level_str = level_chars[self.realm_level - 1] if 1 <= self.realm_level <= 9 else str(self.realm_level)
        return f"{self.realm.value}{level_str}层"


class ItemState(BaseModel):
    """物品状态"""
    name: str
    owner: Optional[str] = None
    location: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class LocationState(BaseModel):
    """地点状态"""
    name: str
    description: str = ""
    unlocked: bool = True
    parent: Optional[str] = None
    flags: Dict[str, Any] = Field(default_factory=dict)


class MapState(BaseModel):
    """地图状态"""
    current: str = ""
    locations: Dict[str, LocationState] = Field(default_factory=dict)
    unlocked_regions: List[str] = Field(default_factory=list)


class WorldState(BaseModel):
    """
    顶层世界状态 - 唯一真相
    
    所有状态变更必须通过 apply_delta() 方法，
    直接修改 state 是禁止的。
    """
    version: str = "2.0"
    revision: int = 0  # 乐观锁版本号
    characters: Dict[str, CharacterState] = Field(default_factory=dict)
    items: Dict[str, ItemState] = Field(default_factory=dict)
    relationships: Dict[str, int] = Field(default_factory=dict)  # "char1|char2": value
    map: MapState = Field(default_factory=MapState)
    global_flags: Dict[str, Any] = Field(default_factory=dict)
    recent_event_ids: List[int] = Field(default_factory=list)  # 最近 100 个事件 ID
    
    # ========== 相变系统新增 ==========
    phase_transitions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="已触发的相变列表"
    )
    
    # ========== 吸引子系统新增 ==========
    attractor_field: Dict[str, Any] = Field(
        default_factory=dict,
        description="叙事引力场配置"
    )
    # =================================
    
    def get_active_characters(self, max_count: int = 20) -> List[str]:
        """按最近活跃时间排序，返回最多 max_count 个角色名"""
        sorted_chars = sorted(
            self.characters.values(),
            key=lambda c: c.last_active,
            reverse=True
        )
        return [c.name for c in sorted_chars[:max_count]]
    
    def get_character(self, name: str) -> Optional[CharacterState]:
        return self.characters.get(name)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorldState':
        """从字典创建 WorldState，并对 hp/mp 进行钳位防止负数"""
        cleaned = data.copy()
        characters = cleaned.get("characters", {})
        for char_data in characters.values():
            if isinstance(char_data, dict):
                if "hp" in char_data and char_data["hp"] < 0:
                    char_data["hp"] = 0
                if "mp" in char_data and char_data["mp"] < 0:
                    char_data["mp"] = 0
                # 可选：对 inventory 排序，保持内部一致性
                if "inventory" in char_data and isinstance(char_data["inventory"], list):
                    char_data["inventory"] = sorted(char_data["inventory"])
        return cls.model_validate(cleaned)
    
    def apply_delta(self, delta: StateDelta) -> 'WorldState':
        """应用状态增量（委托给 delta 的方法）"""
        return delta.apply_to(self)
    
    def get_state_hash(self) -> str:
        return canonical_hash(self.model_dump())
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """序列化，处理 Enum 等，并对所有不关心顺序的字段排序以保证哈希稳定"""
        data = super().model_dump(**kwargs)
        
        # 手动处理特定列表排序（保持现有逻辑）
        if 'characters' in data:
            for char_data in data['characters'].values():
                if 'inventory' in char_data and isinstance(char_data['inventory'], list):
                    char_data['inventory'] = sorted(char_data['inventory'])
                if 'perceived_relationships' in char_data and isinstance(char_data['perceived_relationships'], dict):
                    char_data['perceived_relationships'] = {
                        k: char_data['perceived_relationships'][k]
                        for k in sorted(char_data['perceived_relationships'].keys())
                    }
                if 'relationships' in char_data and isinstance(char_data['relationships'], dict):
                    char_data['relationships'] = {
                        k: char_data['relationships'][k]
                        for k in sorted(char_data['relationships'].keys())
                    }
                if 'flags' in char_data and isinstance(char_data['flags'], dict):
                    char_data['flags'] = {
                        k: char_data['flags'][k]
                        for k in sorted(char_data['flags'].keys())
                    }
        if 'relationships' in data and isinstance(data['relationships'], dict):
            data['relationships'] = {
                k: data['relationships'][k]
                for k in sorted(data['relationships'].keys())
            }
        if 'global_flags' in data and isinstance(data['global_flags'], dict):
            data['global_flags'] = {
                k: data['global_flags'][k]
                for k in sorted(data['global_flags'].keys())
            }
        if 'phase_transitions' in data and isinstance(data['phase_transitions'], list):
            # 按 triggered_at 排序，并对每个字典内部键排序
            data['phase_transitions'] = sorted(
                data['phase_transitions'],
                key=lambda x: x.get('triggered_at', 0)
            )
            # 对每个字典的键排序（通过递归函数处理）
        if 'attractor_field' in data and isinstance(data['attractor_field'], dict):
            if 'attractors' in data['attractor_field']:
                data['attractor_field']['attractors'] = {
                    k: data['attractor_field']['attractors'][k]
                    for k in sorted(data['attractor_field']['attractors'].keys())
                }
        if 'map' in data and isinstance(data['map'], dict):
            if 'locations' in data['map']:
                data['map']['locations'] = {
                    k: data['map']['locations'][k]
                    for k in sorted(data['map']['locations'].keys())
                }
            if 'unlocked_regions' in data['map'] and isinstance(data['map']['unlocked_regions'], list):
                data['map']['unlocked_regions'] = sorted(data['map']['unlocked_regions'])
        if 'items' in data and isinstance(data['items'], dict):
            data['items'] = {
                k: data['items'][k]
                for k in sorted(data['items'].keys())
            }
        
        # 递归规范化所有字典键（确保深层嵌套的字典也有序）
        def normalize(obj):
            if isinstance(obj, dict):
                return {k: normalize(v) for k, v in sorted(obj.items())}
            elif isinstance(obj, list):
                return [normalize(item) for item in obj]
            else:
                return obj
        
        data = normalize(data)
        return data