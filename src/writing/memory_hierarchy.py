"""
三层叙事记忆 - 管理长期状态

L1: Active State（当前活跃状态）
L2: Compressed State（卷级别压缩摘要）
L3: Lore State（永久世界知识）
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class CompressedState(BaseModel):
    volume_num: int
    volume_summary: str = ""
    character_arcs: Dict[str, str] = Field(default_factory=dict)
    key_events: List[str] = Field(default_factory=list)
    resolved_flags: List[str] = Field(default_factory=list)
    major_relationships: Dict[str, int] = Field(default_factory=dict)
    compressed_at: datetime = Field(default_factory=datetime.now)
    character_intents: Dict[str, Any] = Field(default_factory=dict)
    voice_fingerprint: Dict[str, Any] = Field(default_factory=dict)   # 阶段4新增
    narrative_entropy: float = 0.0                                    # 阶段5新增：叙事熵值
    entropy_history: List[float] = Field(default_factory=list)        # 阶段5新增：熵值历史（最近10章）
    # 新增三个字段
    local_entropy: float = 0.0
    arc_entropy: float = 0.0
    civilization_entropy: float = 0.0
    recent_scene_roles: List[str] = Field(default_factory=list)  # 最近20个场景角色标签
    # 认知身份摘要（新增）
    character_identities: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="角色认知身份，格式: {actor: {beliefs, attachments, self_image, moral_boundaries}}"
    )


class LoreState(BaseModel):
    world_rules: List[str] = Field(default_factory=list)
    realm_system: Dict[str, List[str]] = Field(default_factory=dict)
    major_characters: Dict[str, str] = Field(default_factory=dict)
    geography: Dict[str, str] = Field(default_factory=dict)
    cultivation_methods: Dict[str, str] = Field(default_factory=dict)


class NarrativeMemory(BaseModel):
    """三层叙事记忆"""
    active: Optional[Any] = None  # 避免循环导入，使用 Any
    compressed: Dict[int, CompressedState] = Field(default_factory=dict)
    lore: LoreState = Field(default_factory=LoreState)