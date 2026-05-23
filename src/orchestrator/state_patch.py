# src/orchestrator/state_patch.py
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum


class WorkflowPhase(str, Enum):
    """宏观工作流稳定阶段（stable workflow state）"""
    PLANNING = "planning"           # 场景计划已生成，等待写作
    WRITING = "writing"             # 场景正文已写，等待验证
    VALIDATING = "validating"       # 验证完成，等待继续
    TRANSITIONING = "transitioning" # 章节/卷切换中
    COMPLETED = "completed"         # 整部小说完成


@dataclass
class StatePatch:
    """类型安全的状态补丁 - 仅包含 durable workflow state"""
    # 工作流阶段
    phase: Optional[WorkflowPhase] = None
    
    # 进度相关
    current_scene_index: Optional[int] = None
    current_chapter: Optional[int] = None
    current_volume: Optional[int] = None
    total_scenes_in_chapter: Optional[int] = None
    scene_plan_list: Optional[List[Dict[str, Any]]] = None
    scene_plan: Optional[Dict[str, Any]] = None
    
    total_chapters_in_volume: Optional[int] = None
    
    scene_text: Optional[str] = None
    final_answer: Optional[str] = None
    deviation_detected: Optional[bool] = None
    missing_goal_keywords: Optional[List[str]] = None
    missing_conflict_keywords: Optional[List[str]] = None
    
    # 世界状态
    current_state: Optional[Dict[str, Any]] = None
    
    # 跨节点控制
    retry_count: Optional[int] = None
    needs_retry: Optional[bool] = None
    error: Optional[str] = None
    
    # Agent 输出缓存（transient）
    validation_result: Optional[Dict[str, Any]] = None
    writing_feedback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为 LangGraph 可合并的字典（仅非 None 字段）"""
        return {k: v for k, v in self.__dict__.items() if v is not None}