from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime

class KernelEvent(BaseModel):
    id: str
    event_type: str            # STATE_CHANGED, RELATIONSHIP_CHANGED, RESOURCE_CHANGED, KNOWLEDGE_CHANGED
    entity_id: str
    old_value: Any = None
    new_value: Any = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)