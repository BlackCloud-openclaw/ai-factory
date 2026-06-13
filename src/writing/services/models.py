# src/writing/services/models.py
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from src.orchestrator.state_patch import StatePatch
from dataclasses import dataclass, field

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
    character_intents: Optional[Dict[str, Any]] = None   # 新增
    voice_memory: Optional[Dict[str, Any]] = None   # 新增
    
@dataclass
class SceneCompletionResult:
    """场景完成事务的输出 - 只返回事实，不返回路由"""
    state_patch: StatePatch
    chapter_finished: bool = False
    volume_finished: bool = False
    events_applied: int = 0
    error: Optional[str] = None
    

@dataclass
class ScenePlanningCommand:
    """场景计划生成事务的输入"""
    novel_id: str
    volume: int
    chapter: int
    task_type: str
    outline: Optional[Dict[str, Any]]
    current_state: Optional[Dict[str, Any]]
    user_input: str
    resume: bool = False
    # 可选：当前卷的总章节数，可从外部传入避免重复查询
    total_chapters_in_volume: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)  # 新增


@dataclass
class ScenePlanningResult:
    """场景计划生成事务的输出"""
    state_patch: StatePatch
    total_scenes: int = 0
    error: Optional[str] = None
    

@dataclass
class WritingCommand:
    """写作事务的输入"""
    novel_id: str
    volume: int
    chapter: int
    scene_idx: int
    scene_plan: Dict[str, Any]
    current_state: Dict[str, Any]
    writing_feedback: str  # 来自上一次验证失败时的反馈
    # 可选：声纹注册表路径等
    voiceprint_config_path: Optional[str] = None
    # ========== Director 输出字段（阶段2新增）==========
    narrative_blueprint: Optional[Dict[str, Any]] = None
    knowledge_deltas: Optional[List[Dict[str, Any]]] = None
    character_intent: Optional[Dict[str, Any]] = None


@dataclass
class WritingResult:
    """写作事务的输出"""
    state_patch: StatePatch
    scene_text: str
    events: List[Dict[str, Any]]
    deviation_detected: bool = False
    missing_goal_keywords: List[str] = None
    missing_conflict_keywords: List[str] = None
    error: Optional[str] = None