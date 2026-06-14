from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class Knowledge(BaseModel):
    id: str
    holder: str                # 知道该知识的实体
    content: str               # 知识内容
    confidence: float = 1.0    # 确信度
    source: str = ""           # 来源（观测、对话、推理）
    metadata: Dict[str, Any] = Field(default_factory=dict)