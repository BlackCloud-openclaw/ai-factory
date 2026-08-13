"""
Phase 13.1: NarrativeIntent Contract

定义叙事控制平面的核心数据模型。
NarrativeIntent 是 Planner 与 Runtime 之间的协议对象，
描述当前场景在故事状态机中的执行目的。
"""

from enum import Enum
from typing import Optional, List, Any
from pydantic import BaseModel, Field, field_validator
import hashlib


class SceneRole(str, Enum):
    """
    场景角色 - 叙事状态机节点类型。
    每个场景在故事中承担特定的戏剧功能，而非任意的风格标签。
    """

    SETUP = "setup"
    TRANSITION = "transition"
    DISCOVERY = "discovery"
    CONFLICT_ESCALATION = "conflict_escalation"
    CONFRONTATION = "confrontation"
    CHARACTER_DECISION = "character_decision"
    CONSEQUENCE = "consequence"
    RECOVERY = "recovery"
    CLIMAX_PREPARATION = "climax_preparation"
    CLIMAX = "climax"
    RESOLUTION = "resolution"


class NarrativeCondition(BaseModel):
    """
    叙事条件 - 用于 preconditions 的结构化表示。
    允许 Runtime 验证前提是否满足，而非依赖字符串匹配。
    """

    target: str = Field(..., description="条件目标，如 'knowledge.sect_secret'")
    operator: str = Field(..., description="运算符: exists, equals, gt, lt, contains, not_exists")
    expected: Any = Field(..., description="期望值")

    @field_validator("operator")
    @classmethod
    def validate_operator(cls, v: str) -> str:
        allowed = {"exists", "equals", "gt", "lt", "contains", "not_exists"}
        if v not in allowed:
            raise ValueError(f"operator must be one of {allowed}")
        return v


class InteractionPlan(BaseModel):
    """角色交互计划 - Phase 13.3 扩展点，当前仅提供稳定接口"""

    participants: List[str] = Field(default_factory=list, description="参与交互的角色列表")
    relationship_changes: List[str] = Field(default_factory=list, description="关系变化描述")
    conflict: Optional[str] = Field(None, description="交互中的核心冲突")
    emotional_shift: Optional[str] = Field(None, description="期望的情绪变化")


class NarrativeConsequence(BaseModel):
    """
    叙事后果 - 场景结束后必须发生的状态变更。
    连接 NarrativeIntent 与 WorldState 变更的关键桥梁。
    """

    target: str = Field(..., description="状态变更目标")
    operation: str = Field(..., description="操作类型: increase, decrease, set, append, remove")
    value: Any = Field(..., description="变更值")
    event_type: Optional[str] = Field(None, description="对应的 NarrativeEvent 类型")

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, v: str) -> str:
        allowed = {"increase", "decrease", "set", "append", "remove"}
        if v not in allowed:
            raise ValueError(f"operation must be one of {allowed}")
        return v


class NarrativeIntent(BaseModel):
    """
    叙事意图 - 场景执行协议。
    Planner 生成此对象，Runtime 验证并传递给 Writer。
    intent_id 由 Runtime 基于确定性规则生成，不由 LLM 提供。
    """

    intent_id: str = Field(..., description="唯一标识，由 Runtime 生成（非 LLM）")
    scene_role: SceneRole = Field(..., description="场景角色")
    objective: str = Field(..., min_length=5, description="戏剧任务")
    preconditions: List[NarrativeCondition] = Field(default_factory=list)
    beats: List[str] = Field(default_factory=list)
    consequences: List[NarrativeConsequence] = Field(default_factory=list)
    interaction_plan: Optional[InteractionPlan] = Field(None)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> "NarrativeIntent":
        return cls.model_validate(data)

    @classmethod
    def generate_intent_id(cls, scene_id: str, role: SceneRole, objective: str) -> str:
        """确定性生成 intent_id，确保相同输入产生相同 ID。"""
        raw = f"{scene_id}|{role.value}|{objective}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class NarrativeContext(BaseModel):
    """
    Writer 消费的轻量级上下文。
    从 NarrativeIntent 提取 Writer 实际需要的信息，避免暴露完整控制协议。
    """

    scene_role: SceneRole = Field(..., description="场景角色")
    objective: str = Field(..., description="戏剧任务")
    beats: List[str] = Field(default_factory=list, description="必须发生的节拍")
    consequences: List[NarrativeConsequence] = Field(default_factory=list, description="预期的状态变更")

    @classmethod
    def from_intent(cls, intent: NarrativeIntent) -> "NarrativeContext":
        return cls(
            scene_role=intent.scene_role,
            objective=intent.objective,
            beats=intent.beats,
            consequences=intent.consequences,
        )

    def to_prompt_instructions(self) -> str:
        """生成 Writer Prompt 指令"""
        lines = ["## 🎯 叙事意图约束（Narrative Intent）"]

        role_meaning = {
            SceneRole.SETUP: "铺垫信息，建立预期",
            SceneRole.TRANSITION: "场景间过渡",
            SceneRole.DISCOVERY: "信息发现",
            SceneRole.CONFLICT_ESCALATION: "冲突升级",
            SceneRole.CONFRONTATION: "正面对抗",
            SceneRole.CHARACTER_DECISION: "角色关键选择",
            SceneRole.CONSEQUENCE: "后果展示",
            SceneRole.RECOVERY: "恢复/喘息",
            SceneRole.CLIMAX_PREPARATION: "高潮铺垫",
            SceneRole.CLIMAX: "高潮",
            SceneRole.RESOLUTION: "解决/收束",
        }
        meaning = role_meaning.get(self.scene_role, "未定义角色")
        lines.append(f"- **场景角色**: {self.scene_role.value}（{meaning}）")
        lines.append(f"- **戏剧目标**: {self.objective}")

        if self.beats:
            lines.append("- **必须发生的节拍序列**:")
            for beat in self.beats:
                lines.append(f"  - {beat}")

        if self.consequences:
            lines.append("- **预期状态变更（请在事件中体现）**:")
            for c in self.consequences:
                lines.append(f"  - {c.target} {c.operation} {c.value}")

        return "\n".join(lines)