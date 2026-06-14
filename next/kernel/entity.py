from typing import Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field

class EntityType(str, Enum):
    CHARACTER = "character"
    FACTION = "faction"
    LOCATION = "location"
    ITEM = "item"
    CONCEPT = "concept"

class Entity(BaseModel):
    id: str
    name: str
    type: EntityType
    attributes: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)