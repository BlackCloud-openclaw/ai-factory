from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from .entity import Entity
from .capability import Capability
from .relation import Relation
from .knowledge import Knowledge
from .constraint import Constraint

class KernelWorldState(BaseModel):
    """Kernel 世界状态 - 唯一真相的通用表示"""
    version: str = "1.0"
    entities: Dict[str, Entity] = Field(default_factory=dict)
    capabilities: Dict[str, Capability] = Field(default_factory=dict)  # key: "entity_id|capability_name"
    relations: Dict[str, Relation] = Field(default_factory=dict)
    knowledge: Dict[str, Knowledge] = Field(default_factory=dict)
    constraints: Dict[str, Constraint] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)