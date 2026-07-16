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


# ========== v2.1 Scene Specification ==========
class WorldSpec(BaseModel):
    location: str = Field(..., description="地点名称")
    time: Literal["清晨", "正午", "黄昏", "子夜", "深夜"] = Field(..., description="时间")
    atmosphere: str = Field(..., description="氛围关键词")
    sensory: List[str] = Field(..., description="感官细节列表，至少2个")


class EmotionalArc(BaseModel):
    begin: str = Field(..., description="开头情绪")
    middle: str = Field(..., description="中间情绪")
    end: str = Field(..., description="结尾情绪")


class SceneSpecification(BaseModel):
    world: WorldSpec
    mood: str = Field("neutral", description="保留字段，实验中未使用")
    pacing: Literal["slow", "medium", "fast"] = Field(..., description="叙事节奏")
    pov: str = Field(..., description="视角角色")
    emotional_arc: EmotionalArc
    scene_function: Literal[
        "introduce_mystery", "escalate", "release_tension",
        "reveal_truth", "transition", "foreshadow"
    ] = Field(..., description="场景叙事功能")

    def get_function_meaning(self) -> str:
        meanings = {
            "introduce_mystery": "留下谜团，不要在这个场景中给出答案。结尾应产生悬念或疑问。",
            "escalate": "提升冲突烈度，让局势更紧张，压力增大。",
            "release_tension": "缓解紧张情绪，提供情感喘息空间，让读者放松。",
            "reveal_truth": "揭示关键信息，让读者感到震惊或恍然大悟。",
            "transition": "自然过渡，平稳衔接前后情节，节奏舒缓。",
            "foreshadow": "埋下伏笔，暗示未来事件，但不要明说。",
        }
        return meanings.get(self.scene_function, "推进场景叙事。")


class PlanningContract(BaseModel):
    version: str = CONTRACT_VERSION
    scene_id: str
    intent: Intent
    execution: Execution = Field(default_factory=Execution)
    observables: Observables = Field(default_factory=Observables)
    constraints: List[Constraint] = Field(default_factory=list)
    metadata: ContractMetadata
    scene_spec: Optional[SceneSpecification] = None  # v2.1 新增
    
class SceneSpecification(BaseModel):
    world: Dict[str, Any]
    reader_emotion: Dict[str, str]
    narrative_function: str
    pov: str

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