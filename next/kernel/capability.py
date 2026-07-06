from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
from enum import Enum

class CapabilityMode(str, Enum):
    DISCRETE = "discrete"      # 离散等级（如境界）
    CONTINUOUS = "continuous"  # 连续值（如好感度、血量）
    SET = "set"                # 集合（如拥有的物品）

class Capability(BaseModel):
    name: str                  # "cultivation", "hp", "inventory"
    mode: CapabilityMode
    value: Any                 # 离散：字符串等级；连续：float；集合：List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)