from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum

class ConstraintType(str, Enum):
    OATH = "oath"
    CONTRACT = "contract"
    RULE = "rule"
    WORLD_LAW = "world_law"

class Constraint(BaseModel):
    id: str
    type: ConstraintType
    description: str
    owner: str                 # 角色、势力、或 "world"
    target: Optional[str] = None
    severity: float = 1.0
    is_active: bool = True
    expires_at: Optional[int] = None  # 章节号
    metadata: Dict[str, Any] = Field(default_factory=dict)