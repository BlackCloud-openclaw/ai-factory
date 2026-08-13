from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class JudgeContext:
    """LLM Judge 专用上下文，与 EvaluationContext 隔离。"""
    previous_scene_text: Optional[str] = None
    character_summary: Optional[str] = None
    world_summary: Optional[str] = None
    dialogue_history: Optional[str] = None
    chapter_summary: Optional[str] = None
    volume_summary: Optional[str] = None

    extra: Dict[str, Any] = field(default_factory=dict)