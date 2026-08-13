# src/writing/services/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from src.orchestrator.state_patch import StatePatch
from src.narrative.intent import IntentResolver
from src.writing.narrative_intent import NarrativeIntent


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
    character_intents: Optional[Dict[str, Any]] = None
    voice_memory: Optional[Dict[str, Any]] = None
    raw_output: Optional[str] = None
    narrative_intent: Optional[NarrativeIntent] = None


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
    total_chapters_in_volume: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    intent_resolver: Optional[IntentResolver] = None


@dataclass
class ScenePlanningResult:
    """
    场景计划生成事务的输出。

    核心契约：
    - state_patch: 通用状态修改
    - total_scenes: 本场景计划包含的场景总数
    - planner_outputs: 每个场景的 PlannerOutput（含 NarrativeIntent + ExecutionContract）
      这是 Writer / Validator 消费的核心业务产物，必须通过显式契约传递。
    - error: 失败传播
    """
    state_patch: StatePatch
    total_scenes: int = 0
    planner_outputs: List[Dict[str, Any]] = field(default_factory=list)  # Phase 13.2.2 新增
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
    writing_feedback: str
    voiceprint_config_path: Optional[str] = None
    narrative_blueprint: Optional[Dict[str, Any]] = None
    knowledge_deltas: Optional[List[Dict[str, Any]]] = None
    character_intent: Optional[Dict[str, Any]] = None
    drama_structure: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    # D.4.1-a: 显式契约传输
    execution_contract: Optional[Dict[str, Any]] = None


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