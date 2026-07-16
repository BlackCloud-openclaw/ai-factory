# experiments/phase4/planning_contract_local.py
"""
Planning Contract v1.0 本地定义（扩展 v2.1 Scene Specification）
用于 Phase 4 实验，不依赖主代码。
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime

CONTRACT_VERSION = "1.0"


class ExecutionUnit(BaseModel):
    id: str = Field(..., description="单元唯一标识")
    label: Literal["action", "beat", "intent", "constraint"] = Field(..., description="单元类型")
    description: str = Field(..., description="自然语言描述")
    attributes: Dict[str, Any] = Field(default_factory=dict)


class Execution(BaseModel):
    units: List[ExecutionUnit] = Field(default_factory=list)


class Constraint(BaseModel):
    type: Literal["required", "forbidden", "before", "after", "exclusive", "at_least_once"]
    target: str
    condition: Optional[str] = None
    refs: Optional[List[str]] = None


class StateChange(BaseModel):
    type: Literal["plot_flag", "relationship", "inventory", "realm", "location", "hp"]
    name: Optional[str] = None
    value: Optional[Any] = None
    from_char: Optional[str] = None
    to_char: Optional[str] = None
    delta: Optional[int] = None
    actor: Optional[str] = None
    item: Optional[str] = None
    operation: Optional[Literal["acquire", "lose"]] = None
    quantity: Optional[int] = 1
    to_major_realm: Optional[str] = None
    to_minor_stage: Optional[int] = None
    location: Optional[str] = None
    new_hp: Optional[int] = None


class Observables(BaseModel):
    state_changes: List[StateChange] = Field(default_factory=list)
    story_events: List[Dict] = Field(default_factory=list)
    narrative_flags: List[Dict] = Field(default_factory=list)


class Intent(BaseModel):
    goal: str
    conflict: str
    expected_outcome: str


class ContractMetadata(BaseModel):
    chapter: int
    scene_index: int
    arc: Optional[str] = None
    created_at: Optional[datetime] = Field(default_factory=datetime.now)

# ========== v2.1: Scene Specification ==========

class WorldSpec(BaseModel):
    location: str = Field(..., description="地点名称")
    time: str = Field(..., description="时间")
    atmosphere: str = Field(..., description="氛围关键词")
    sensory: List[str] = Field(default_factory=list, description="感官细节列表")


class EmotionalArc(BaseModel):
    begin: str = Field(..., description="开头情绪")
    middle: str = Field(..., description="中间情绪")
    end: str = Field(..., description="结尾情绪")


class SceneSpecification(BaseModel):
    """
    v2.1 场景规格 - 控制读者体验
    """
    world: WorldSpec = Field(..., description="世界事实")
    mood: str = Field(default="neutral", description="整体基调（保留兼容）")
    pacing: str = Field(default="medium", description="叙事节奏（保留兼容）")
    reader_emotion: EmotionalArc = Field(..., description="读者情绪轨迹")
    narrative_function: str = Field(..., description="叙事功能")
    pov: str = Field(..., description="视角角色名")

    def get_function_meaning(self) -> str:
        meanings = {
            "introduce_mystery": "留下谜团，不给出答案，结尾产生悬念",
            "escalate": "提升冲突，压力增大，局势紧张",
            "reveal_truth": "揭示关键信息，让读者震惊",
            "release_tension": "缓解紧张，提供喘息空间",
            "transition": "自然过渡，节奏平稳",
            "foreshadow": "埋下伏笔，暗示未来事件",
        }
        return meanings.get(self.narrative_function, "推进场景叙事")

class PlanningContract(BaseModel):
    version: str = CONTRACT_VERSION
    scene_id: str
    intent: Intent
    execution: Execution = Field(default_factory=Execution)
    observables: Observables = Field(default_factory=Observables)
    constraints: List[Constraint] = Field(default_factory=list)
    metadata: ContractMetadata
    scene_spec: Optional[SceneSpecification] = None  # v2.1 新增
    
