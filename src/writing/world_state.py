"""
规范化的世界状态模型 - 唯一真状态
"""
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, validator
from .delta import StateDelta
from src.common.canonical import canonical_hash

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
    """角色状态"""
    name: str
    realm: Realm = Realm.REFINING_QI
    realm_level: int = Field(1, ge=1, le=9)
    hp: int = Field(100, ge=0)
    mp: int = Field(100, ge=0)
    inventory: List[str] = Field(default_factory=list)
    relationships: Dict[str, int] = Field(default_factory=dict)  # "target_char": value (-100..100)
    location: str = ""
    flags: Dict[str, Any] = Field(default_factory=dict)
    last_active: datetime = Field(default_factory=datetime.now)
    
    @validator('realm_level')
    def level_within_bounds(cls, v, values):
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
        # 复制数据避免修改原字典
        cleaned = data.copy()
        characters = cleaned.get("characters", {})
        for char_data in characters.values():
            if isinstance(char_data, dict):
                if "hp" in char_data and char_data["hp"] < 0:
                    char_data["hp"] = 0
                if "mp" in char_data and char_data["mp"] < 0:
                    char_data["mp"] = 0
        return cls.model_validate(cleaned)    

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """序列化，处理 Enum 等"""
        return super().model_dump(**kwargs)
    
    def apply_delta(self, delta: StateDelta) -> 'WorldState':
        """应用状态增量（委托给 delta 的方法）"""
        return delta.apply_to(self)
    
    def get_state_hash(self) -> str:
        return canonical_hash(self.model_dump())