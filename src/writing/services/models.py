# src/writing/services/models.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
from src.orchestrator.state_patch import StatePatch


@dataclass
class SceneCompletionCommand:
    """场景完成事务的输入"""
    novel_id: str
    volume: int
    chapter: int
    scene_idx: int
    total_scenes: int
    current_world_state: Dict[str, Any]
    parsed_output: Dict[str, Any]      # 包含 events, scene_text
    scene_plan: Optional[Dict[str, Any]] = None


@dataclass
class SceneCompletionResult:
    """场景完成事务的输出 - 只返回事实，不返回路由"""
    state_patch: StatePatch
    chapter_finished: bool = False
    volume_finished: bool = False
    events_applied: int = 0
    error: Optional[str] = None