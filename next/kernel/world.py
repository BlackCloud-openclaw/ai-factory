# next/kernel/world.py
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from .entity import Entity
from .capability import Capability
from .relation import Relation
from .knowledge import Knowledge
from .constraint import Constraint

class KernelWorldState(BaseModel):
    version: str = "1.0"
    entities: Dict[str, Entity] = Field(default_factory=dict)
    capabilities: Dict[str, Capability] = Field(default_factory=dict)
    relations: Dict[str, Relation] = Field(default_factory=dict)
    knowledge: Dict[str, Knowledge] = Field(default_factory=dict)
    constraints: Dict[str, Constraint] = Field(default_factory=dict)
    # 新增：相变历史和吸引子配置
    phase_transitions: List[Dict[str, Any]] = Field(default_factory=list)
    attractor_field: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)