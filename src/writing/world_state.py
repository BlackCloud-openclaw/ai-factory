"""
规范化的世界状态模型 - 唯一真状态

Phase 4C 变更：
- CharacterState 增加 `id` 字段（稳定标识）
- model_dump() 序列化时 characters 字典键使用 `id`（而非 name）
- from_dict() 加载时自动为角色补全 `id` 字段
- 新增 get_character() 等 API（已在 Phase 4A 添加）
"""
import logging
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
from .delta import StateDelta
from src.common.canonical import canonical_hash
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from src.writing.constraint import Constraint
from src.domain.identity import get_character_id_by_name, get_character_config

# 创建 logger
logger = logging.getLogger(__name__)


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
    # ========== Phase 4C 新增：稳定 ID ==========
    id: Optional[str] = Field(
        default=None,
        description="角色稳定标识（如 protagonist, mentor），由配置定义"
    )
    # =============================================
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

    # ========== 认知身份 ==========
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
    revision: int = 0
    characters: Dict[str, CharacterState] = Field(default_factory=dict)  # key = character_id
    items: Dict[str, ItemState] = Field(default_factory=dict)
    relationships: Dict[str, int] = Field(default_factory=dict)
    map: MapState = Field(default_factory=MapState)
    global_flags: Dict[str, Any] = Field(default_factory=dict)
    recent_event_ids: List[int] = Field(default_factory=list)

    # ========== 相变系统 ==========
    phase_transitions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="已触发的相变列表"
    )

    # ========== 吸引子系统 ==========
    attractor_field: Dict[str, Any] = Field(
        default_factory=dict,
        description="叙事引力场配置"
    )

    # ========== 全局约束系统 ==========
    constraints: List[Constraint] = Field(
        default_factory=list,
        description="全局约束列表（誓言、契约、规则）"
    )

    # =========================================================
    # Phase 4A / 4B: 角色访问 API
    # =========================================================
    def get_character(self, key: str) -> Optional[CharacterState]:
        """
        获取角色，支持 ID 或名称作为 key
        ID 优先，名称回退
        """
        if not key:
            return None

        # 1. 直接用 key 查找（期望是 ID）
        if key in self.characters:
            return self.characters[key]

        # 2. 尝试将 key 作为名称，查找对应的 ID
        char_id = get_character_id_by_name(key)
        if char_id and char_id in self.characters:
            return self.characters[char_id]

        # 3. 遍历查找（兜底）
        for char in self.characters.values():
            if char.name == key:
                return char
            if hasattr(char, 'id') and char.id == key:
                return char

        return None

    def get_character_by_id(self, char_id: str) -> Optional[CharacterState]:
        """通过 ID 获取角色（推荐）"""
        return self.characters.get(char_id)

    def get_character_by_name(self, name: str) -> Optional[CharacterState]:
        """通过名称获取角色（兼容）"""
        return self.get_character(name)

    def get_all_characters(self) -> List[CharacterState]:
        """获取所有角色（仅遍历一次）"""
        return list(self.characters.values())

    def get_all_character_ids(self) -> List[str]:
        """获取所有角色 ID（不含名称）"""
        return list(self.characters.keys())

    def get_all_character_names(self) -> List[str]:
        """获取所有角色显示名称（用于显示层）"""
        return [char.name for char in self.characters.values()]

    def get_character_count(self) -> int:
        """返回实际角色数量（ID 数量）"""
        return len(self.characters)

    # =========================================================
    # Phase 4C: 序列化/反序列化迁移
    # =========================================================
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """
        序列化时，将 characters 字典的键从 name 转换为 ID
        确保存储格式统一为 ID-key
        """
        data = super().model_dump(**kwargs)

        # ---------- 1. 转换 characters 键 ----------
        if 'characters' in data and data['characters']:
            id_based = {}
            config = get_character_config()
            for old_key, char_data in data['characters'].items():
                # 尝试从 char_data 中提取 ID
                char_id = None
                if isinstance(char_data, dict):
                    char_id = char_data.get('id')
                elif hasattr(char_data, 'id'):
                    char_id = getattr(char_data, 'id', None)

                if char_id:
                    id_based[char_id] = char_data
                else:
                    # 从配置中通过名称查找 ID
                    name = char_data.get('name') if isinstance(char_data, dict) else getattr(char_data, 'name', None)
                    if name:
                        char_id = config.get_character_id_by_name(name)
                    if char_id:
                        id_based[char_id] = char_data
                    else:
                        # 最后保留原键
                        id_based[old_key] = char_data
            data['characters'] = id_based

        # ---------- 2. 排序（保持确定性哈希） ----------
        if 'characters' in data:
            for char_data in data['characters'].values():
                if isinstance(char_data, dict):
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
            data['phase_transitions'] = sorted(
                data['phase_transitions'],
                key=lambda x: x.get('triggered_at', 0)
            )
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
        if 'constraints' in data and isinstance(data['constraints'], list):
            data['constraints'] = sorted(data['constraints'], key=lambda x: x.get('id', ''))

        # 递归规范化所有字典键
        def normalize(obj):
            if isinstance(obj, dict):
                return {k: normalize(v) for k, v in sorted(obj.items())}
            elif isinstance(obj, list):
                return [normalize(item) for item in obj]
            else:
                return obj

        data = normalize(data)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorldState':
        """
        从字典创建 WorldState，钳位 hp/mp 并确保角色有 id 字段
        """
        cleaned = data.copy()
        characters = cleaned.get("characters", {})
        config = get_character_config()

        # ---------- 1. 确保每个角色都有 id ----------
        for key, char_data in characters.items():
            if isinstance(char_data, dict):
                # 如果已有 id 且有效，保留
                if 'id' in char_data and char_data['id']:
                    pass
                # 否则尝试从 key 或 name 推断
                else:
                    # 尝试从 key 获取 ID（如果 key 是 ID）
                    if config.get_character(key) is not None:
                        char_data['id'] = key
                    # 否则从 name 查找
                    elif 'name' in char_data and char_data['name']:
                        char_id = config.get_character_id_by_name(char_data['name'])
                        if char_id:
                            char_data['id'] = char_id
                    # 最后兜底：使用 key 作为 id
                    if 'id' not in char_data or not char_data['id']:
                        char_data['id'] = key

                # 钳位 hp/mp
                if "hp" in char_data and char_data["hp"] < 0:
                    char_data["hp"] = 0
                if "mp" in char_data and char_data["mp"] < 0:
                    char_data["mp"] = 0
                if "inventory" in char_data and isinstance(char_data["inventory"], list):
                    char_data["inventory"] = sorted(char_data["inventory"])

        # ---------- 2. 构建 WorldState 实例 ----------
        # 注意：如果 characters 的键不是 ID，Pydantic 仍然接受，但 model_dump 会转换
        # 这里我们保留原样，由 model_dump 统一处理
        return cls.model_validate(cleaned)

    # =========================================================
    # 原有方法（未改动）
    # =========================================================
    def get_active_characters(self, max_count: int = 20) -> List[str]:
        sorted_chars = sorted(
            self.characters.values(),
            key=lambda c: c.last_active,
            reverse=True
        )
        return [c.name for c in sorted_chars[:max_count]]

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def apply_delta(self, delta: StateDelta) -> 'WorldState':
        return delta.apply_to(self)

    def get_state_hash(self) -> str:
        return canonical_hash(self.model_dump())

    def add_constraint(self, constraint: Constraint):
        self.constraints.append(constraint)

    def get_constraints_for(self, owner: str) -> List[Constraint]:
        return [c for c in self.constraints if c.owner == owner and c.is_active]