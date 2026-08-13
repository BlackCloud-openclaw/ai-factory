from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class SceneExecutionContext:
    """场景执行的静态上下文，不可变"""
    chapter_id: str
    scene_id: str
    scene_role: str
    dramatic_function: str
    characters: List[str] = field(default_factory=list)
    location: str = ""
    time: str = ""