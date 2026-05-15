"""
三层叙事记忆 - 管理长期状态

L1: Active State（当前活跃状态）
L2: Compressed State（卷级别压缩摘要）
L3: Lore State（永久世界知识）
"""
from typing import Dict, List, Optional
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


class LoreState(BaseModel):
    world_rules: List[str] = Field(default_factory=list)
    realm_system: Dict[str, List[str]] = Field(default_factory=dict)
    major_characters: Dict[str, str] = Field(default_factory=dict)
    geography: Dict[str, str] = Field(default_factory=dict)
    cultivation_methods: Dict[str, str] = Field(default_factory=dict)


class NarrativeMemory(BaseModel):
    """三层叙事记忆"""
    active: 'WorldState' = None  # 延迟导入，避免循环
    compressed: Dict[int, CompressedState] = Field(default_factory=dict)
    lore: LoreState = Field(default_factory=LoreState)