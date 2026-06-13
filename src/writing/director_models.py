# src/writing/director_models.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class SceneRole(str, Enum):
    """场景角色 - 控制节奏和余波"""
    SETUP = "SETUP"           # 铺垫信息，建立预期
    ESCALATION = "ESCALATION" # 提升张力，增加压力
    REVEAL = "REVEAL"         # 关键信息曝光
    RELEASE = "RELEASE"       # 张力释放
    AFTERMATH = "AFTERMATH"   # 情绪沉淀，后果展示
    TRANSITION = "TRANSITION" # 场景间衔接


@dataclass
class NarrativeBlueprint:
    """叙事蓝图 - Director 的唯一输出"""
    attention_path: List[str] = field(default_factory=list)   # 主角注意力轨迹
    withheld_information: str = ""                           # 被延迟的信息
    reveal_beat: str = ""                                    # 情绪变化瞬间
    scene_pressure: str = ""                                 # 压力来源与可见性
    silent_action_priority: str = ""                         # 哪个动作比对白更重要
    recurring_image: str = ""                                # 反复出现的意象
    scene_role: SceneRole = SceneRole.SETUP                  # 场景角色


@dataclass
class KnowledgeDelta:
    """知识变化 - 控制信息释放"""
    holder: str
    information: str
    operation: str       # acquire / lose / doubt / confirm
    trigger: str
    visibility: str      # reader_visible / hidden
    source: str = ""     # 信息来源（如 "偷听", "古籍", "推理"）
    reliability: float = 1.0  # 0.0-1.0，默认完全可信


@dataclass
class CharacterIntent:
    """角色意图 - 只读，由 Planner 或 Director 生成"""
    actor: str
    conscious_goal: str          # 显性目标
    hidden_need: str             # 深层需求
    fear: str                    # 恐惧什么
    misconception: Optional[str] = None
    immediate_tactic: str = ""   # 具体行动方式
    perceived_relationships: Optional[Dict[str, Dict[str, Any]]] = None
    
    # ========== 认知身份（新增）==========
    # 允许 Director 微调，但不能完全违背现有身份
    beliefs: Optional[List[str]] = None          # 核心信念
    attachments: Optional[List[str]] = None      # 依恋
    self_image: Optional[str] = None             # 自我认知
    moral_boundaries: Optional[List[str]] = None # 道德底线
    identity_change_reason: Optional[str] = None # 如果身份变化，说明原因
    # ===================================


@dataclass
class SceneSkeleton:
    """场景骨架 - 由 Planner 生成，Director 和 Writer 只读"""
    goal: str
    conflict: str
    must_events: List[str] = field(default_factory=list)
    state_delta: Dict[str, Any] = field(default_factory=dict)
    arc_progress: Dict[str, Any] = field(default_factory=dict)
    scene_objective: str = ""   # 新增：场景存在的理由