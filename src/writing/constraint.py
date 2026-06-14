from pydantic import BaseModel
from typing import Optional, List
from enum import Enum
from datetime import datetime

class ConstraintType(str, Enum):
    OATH = "oath"           # 誓言
    CONTRACT = "contract"   # 契约
    RULE = "rule"           # 规则/禁令
    WORLD_LAW = "world_law" # 世界法则

class Constraint(BaseModel):
    id: str
    type: ConstraintType
    description: str
    owner: str              # 角色名或 "world"
    target: Optional[str] = None   # 约束对象（角色、势力、地点）
    severity: float = 1.0   # 违背严重程度
    expires_at: Optional[int] = None  # 章节数（绝对或相对）
    created_at: datetime = datetime.now()
    is_active: bool = True

class ConstraintRegistry:
    """约束注册表 - 全局存储和管理约束"""
    def __init__(self, constraints: List[Constraint] = None):
        self.constraints: List[Constraint] = constraints or []
    
    def add(self, constraint: Constraint):
        self.constraints.append(constraint)
    
    def remove(self, constraint_id: str):
        self.constraints = [c for c in self.constraints if c.id != constraint_id]
    
    def get_for_owner(self, owner: str) -> List[Constraint]:
        return [c for c in self.constraints if c.owner == owner and c.is_active]
    
    def get_active(self) -> List[Constraint]:
        return [c for c in self.constraints if c.is_active]
    
    def check_violation(self, event_type: str, actor: str, target: Optional[str] = None) -> List[Constraint]:
        """检查事件是否违反任何活跃约束"""
        violated = []
        for c in self.get_active():
            if c.owner != actor and c.owner != "world":
                continue
            if c.target and target and c.target != target:
                continue
            # 简单关键词匹配（可根据需要扩展）
            if c.description in event_type or any(kw in event_type for kw in c.description.split()):
                violated.append(c)
        return violated
