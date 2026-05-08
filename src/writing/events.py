# src/writing/events.py
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel

class Event(BaseModel):
    event_id: str
    sequence_id: int = 0          # 数据库自增，0表示未持久化
    type: str
    payload: Dict[str, Any]
    created_at: datetime
    novel_id: str
    chapter_id: Optional[str] = None

    @classmethod
    def new(cls, event_type: str, payload: Dict, novel_id: str, chapter_id: Optional[str] = None):
        return cls(
            event_id=str(uuid.uuid4()),
            type=event_type,
            payload=payload,
            created_at=datetime.utcnow(),
            novel_id=novel_id,
            chapter_id=chapter_id
        )

# 事件类型常量
EVENT_CHARACTER_UPDATE = "character_update"
EVENT_TIMELINE_ADD = "timeline_add"
EVENT_WORLD_RULE_ADD = "world_rule_add"
EVENT_CHAPTER_FINISHED = "chapter_finished"