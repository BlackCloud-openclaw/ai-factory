from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class Relation(BaseModel):
    id: str
    from_entity: str
    to_entity: str
    relation_type: str         # "friendship", "hostility", "kinship", "ownership"
    value: float = 0.0         # -100..100
    confidence: float = 1.0    # 0..1
    metadata: Dict[str, Any] = Field(default_factory=dict)