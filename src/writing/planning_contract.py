"""
Planning Contract v1.0 - AI Factory 稳定规划接口（Stable Planning Interface）

Planning Contract 是 AI Factory v2.0 的核心抽象，定义了 Planner 对每个场景的承诺（Promise）。
它是一个与 Runtime 无关的领域 DSL，所有 Agent（Planner、Writer、Validator）都围绕此契约工作。

核心设计原则：
- Contract 是冻结的接口，允许 Planner/Writer/Validator/Runtime 独立演进
- 不包含任何 Runtime 概念（Event、Projection、Operation）
- 不包含任何实验变量（representation、density、temperature）
- 从第一天支持版本迁移（ContractUpcaster）
"""

from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from datetime import datetime


# ============================================================================
# 版本定义
# ============================================================================

CONTRACT_VERSION = "1.0"


# ============================================================================
# Execution Unit - 执行单元
# ============================================================================

class ExecutionUnit(BaseModel):
    """
    执行单元 - Contract 中最核心的结构化元素。
    
    每种规划表示（Action/Beat/Intent/Constraint）都使用统一的 ExecutionUnit，
    通过 label 区分类型，通过 attributes 承载结构化信息。
    
    Attributes:
        id: 单元唯一标识（场景内）
        label: 单元类型标签（action | beat | intent | constraint）
        description: 自然语言描述
        attributes: 结构化属性（可选的键值对）
    """
    id: str = Field(..., description="单元唯一标识，如 'A1', 'B2'")
    label: Literal["action", "beat", "intent", "constraint"] = Field(
        ..., description="单元类型：action | beat | intent | constraint"
    )
    description: str = Field(..., description="自然语言描述，如 '大师兄踩住碎片'")
    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="结构化属性，如 {'actor': '大师兄', 'target': '碎片'}"
    )

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """序列化时确保 id 始终为字符串"""
        data = super().model_dump(**kwargs)
        data["id"] = str(data["id"])
        return data


# ============================================================================
# Execution - 执行计划
# ============================================================================

class Execution(BaseModel):
    """
    执行计划 - 描述场景如何完成。
    
    units 是 ExecutionUnit 的列表，表示一个有序或无序的执行单元集合。
    不同规划表示（Action/Beat/Intent/Constraint）的区别体现在 units 的内容和 label 上。
    """
    units: List[ExecutionUnit] = Field(
        default_factory=list,
        description="执行单元列表"
    )


# ============================================================================
# Constraint - 约束
# ============================================================================

class Constraint(BaseModel):
    """
    结构化约束 - 替代自然语言的 must_happen / must_not_happen。
    
    支持多种约束类型，便于 Validator 进行精确验证。
    
    Types:
        - required: 必须发生
        - forbidden: 禁止发生
        - before: 在某个事件之前发生
        - after: 在某个事件之后发生
        - exclusive: 互斥，只能发生其一
        - at_least_once: 至少发生一次
    """
    type: Literal["required", "forbidden", "before", "after", "exclusive", "at_least_once"] = Field(
        ..., description="约束类型"
    )
    target: str = Field(..., description="约束目标描述，如 '大师兄出现'")
    condition: Optional[str] = Field(
        default=None,
        description="可选条件，如 '必须在林逸之前'"
    )
    refs: Optional[List[str]] = Field(
        default=None,
        description="关联的 ExecutionUnit ID 列表"
    )

    @field_validator("target")
    @classmethod
    def validate_target(cls, v: str) -> str:
        if not v or len(v.strip()) < 2:
            raise ValueError("约束目标至少需要2个字符")
        return v.strip()


# ============================================================================
# StateChange - 状态变化（Observable）
# ============================================================================

class StateChange(BaseModel):
    """
    状态变化 - 描述场景结束后世界状态应该发生的变化。
    
    这是 Planning Contract 与 World State 之间的契约。
    """
    type: Literal["plot_flag", "relationship", "inventory", "realm", "location", "hp"] = Field(
        ..., description="状态变化类型"
    )
    # Plot Flag
    name: Optional[str] = Field(None, description="flag 名称（plot_flag 类型必填）")
    value: Optional[Any] = Field(None, description="flag 值（plot_flag 类型必填）")
    # Relationship
    from_char: Optional[str] = Field(None, description="关系发起方（relationship 类型必填）")
    to_char: Optional[str] = Field(None, description="关系接收方（relationship 类型必填）")
    delta: Optional[int] = Field(None, description="关系变化量（relationship 类型必填）")
    # Inventory
    actor: Optional[str] = Field(None, description="操作者（inventory/realm/location/hp 类型必填）")
    item: Optional[str] = Field(None, description="物品名称（inventory 类型必填）")
    operation: Optional[Literal["acquire", "lose"]] = Field(None, description="操作类型（inventory 类型必填）")
    quantity: Optional[int] = Field(1, description="数量（inventory 类型可选）")
    # Realm
    to_major_realm: Optional[str] = Field(None, description="目标大境界（realm 类型必填）")
    to_minor_stage: Optional[int] = Field(None, description="目标小层级（realm 类型必填）")
    # Location
    location: Optional[str] = Field(None, description="目标地点（location 类型必填）")
    # HP
    new_hp: Optional[int] = Field(None, description="新 HP 值（hp 类型必填）")

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        data = super().model_dump(**kwargs)
        # 清理空值字段，仅保留 type 和必要字段
        cleaned = {"type": data["type"]}
        if self.type == "plot_flag":
            cleaned["name"] = data.get("name")
            cleaned["value"] = data.get("value")
        elif self.type == "relationship":
            cleaned["from_char"] = data.get("from_char")
            cleaned["to_char"] = data.get("to_char")
            cleaned["delta"] = data.get("delta")
        elif self.type in ["inventory", "realm", "location", "hp"]:
            for key in ["actor", "item", "operation", "quantity", "to_major_realm", "to_minor_stage", "location", "new_hp"]:
                if data.get(key) is not None:
                    cleaned[key] = data[key]
        return cleaned


# ============================================================================
# StoryEvent - 故事事件（Observable）
# ============================================================================

class StoryEvent(BaseModel):
    """
    故事事件 - 描述场景中应该发生的关键故事事件。
    
    与 StateChange 不同，StoryEvent 关注的是叙事层面的事件，
    而非状态层面的变化。
    """
    type: Literal["dialogue", "discovery", "combat", "decision"] = Field(
        ..., description="事件类型：dialogue | discovery | combat | decision"
    )
    description: str = Field(..., description="事件描述")
    participants: List[str] = Field(
        default_factory=list,
        description="参与角色列表"
    )
    importance: Literal["low", "normal", "high", "critical"] = Field(
        default="normal",
        description="重要性级别"
    )


# ============================================================================
# NarrativeFlag - 叙事标记（Observable）
# ============================================================================

class NarrativeFlag(BaseModel):
    """
    叙事标记 - 描述场景结束后应该建立的叙事标记。
    
    与 PlotFlag 不同，NarrativeFlag 更倾向于叙事层面（如 tension_level、mood），
    而不是状态层面（如 investigation_blocked）。
    """
    name: str = Field(..., description="标记名称")
    value: Any = Field(..., description="标记值")


# ============================================================================
# Observables - 可观测事项
# ============================================================================

class Observables(BaseModel):
    """
    可观测事项 - Planner 对场景结束后世界状态的预测。
    
    这是 Planning Contract 与 World State 之间的核心桥梁。
    """
    state_changes: List[StateChange] = Field(
        default_factory=list,
        description="期望的状态变化列表"
    )
    story_events: List[StoryEvent] = Field(
        default_factory=list,
        description="期望的故事事件列表"
    )
    narrative_flags: List[NarrativeFlag] = Field(
        default_factory=list,
        description="期望的叙事标记列表"
    )


# ============================================================================
# Intent - 故事意图
# ============================================================================

class Intent(BaseModel):
    """
    故事意图 - Planner 对这个场景的叙事目标。
    
    这是整个 Contract 中最接近故事层面的描述。
    """
    goal: str = Field(..., description="场景目标（一句话）")
    conflict: str = Field(..., description="核心冲突（一句话）")
    expected_outcome: str = Field(..., description="预期结果（一句话）")

    @field_validator("goal", "conflict", "expected_outcome")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or len(v.strip()) < 3:
            raise ValueError("字段至少需要3个字符")
        return v.strip()


# ============================================================================
# Metadata - 领域元数据
# ============================================================================

class ContractMetadata(BaseModel):
    """
    Contract 元数据 - 仅包含故事领域信息，不含实验变量。
    """
    chapter: int = Field(..., description="所属章节")
    scene_index: int = Field(..., description="场景序号（从0开始）")
    arc: Optional[str] = Field(None, description="所属弧线标识")
    created_at: Optional[datetime] = Field(
        default_factory=datetime.now,
        description="创建时间"
    )

    @field_validator("chapter")
    @classmethod
    def validate_chapter(cls, v: int) -> int:
        if v < 1:
            raise ValueError("章号必须大于0")
        return v

    @field_validator("scene_index")
    @classmethod
    def validate_scene_index(cls, v: int) -> int:
        if v < 0:
            raise ValueError("场景序号必须大于等于0")
        return v


# ============================================================================
# v2.1: Scene Specification
# ============================================================================

class SceneSpecification(BaseModel):
    """
    v2.1 场景规格 - 控制读者体验
    
    这是 Empirical Control Model 的工程化产物。
    四个维度（World, Reader Emotion, Narrative Function, POV）全部通过实验验证有效。
    """
    world: Dict[str, Any] = Field(
        ...,
        description="世界事实：location（地点）, time（时间）, atmosphere（氛围）, sensory（感官细节列表）"
    )
    reader_emotion: Dict[str, str] = Field(
        ...,
        description="读者情绪轨迹：begin（开头）, middle（中间）, end（结尾），三者必须不同"
    )
    narrative_function: str = Field(
        ...,
        description="叙事功能：introduce_mystery | escalate | reveal_truth | release_tension | transition | foreshadow"
    )
    pov: str = Field(
        ...,
        description="视角角色名，如'林逸'、'二叔'"
    )

    def get_function_meaning(self) -> str:
        meanings = {
            "introduce_mystery": "留下谜团，不给出答案，结尾产生悬念",
            "escalate": "提升冲突，压力增大，局势紧张",
            "reveal_truth": "揭示关键信息，让读者震惊",
            "release_tension": "缓解紧张，提供喘息空间",
            "transition": "自然过渡，节奏平稳",
            "foreshadow": "埋下伏笔，暗示未来事件",
        }
        return meanings.get(self.narrative_function, "推进叙事")

# ============================================================================
# Planning Contract - 核心契约
# ============================================================================

class PlanningContract(BaseModel):
    """
    Planning Contract v1.0 - AI Factory 稳定规划接口。
    
    这是整个系统的核心抽象，所有 Agent 之间的交互都通过此契约进行。
    
    Attributes:
        version: 契约版本（用于迁移）
        scene_id: 场景唯一标识
        intent: 故事意图
        execution: 执行计划（包含 units）
        observables: 可观测事项（Planner 对世界的预期）
        constraints: 结构化约束
        metadata: 领域元数据
    """
    version: str = Field(default=CONTRACT_VERSION, description="契约版本")
    scene_id: str = Field(..., description="场景唯一标识")
    intent: Intent = Field(..., description="故事意图")
    execution: Execution = Field(default_factory=Execution, description="执行计划")
    observables: Observables = Field(default_factory=Observables, description="可观测事项")
    constraints: List[Constraint] = Field(default_factory=list, description="结构化约束")
    metadata: ContractMetadata = Field(..., description="领域元数据")
    scene_spec: Optional[SceneSpecification] = Field(
        default=None,
        description="v2.1 场景规格（控制读者体验）"
    )

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """序列化时自动包含 version"""
        data = super().model_dump(**kwargs)
        data["version"] = self.version
        return data

class ContractUpcaster:
    """
    Contract 版本迁移器。
    
    确保旧版本 Contract 可以平滑升级到最新版本。
    """
    
# src/writing/planning_contract.py

class ContractUpcaster:
    """Contract 版本迁移器。"""
    
    @staticmethod
    def upcast(data: Dict[str, Any]) -> Dict[str, Any]:
        version = data.get("version", "0.9")
        
        if version == "0.9":
            data = ContractUpcaster._upcast_v0_9_to_v1_0(data)
        
        # ========== v2.1: 保留 scene_spec ==========
        if "scene_spec" in data and data["scene_spec"] is not None:
            # 确保 scene_spec 在结果中
            if "scene_spec" not in data:
                data["scene_spec"] = None
            # 如果 scene_spec 是字典，保留它
            if isinstance(data["scene_spec"], dict):
                data["scene_spec"] = data["scene_spec"]
        
        data["version"] = CONTRACT_VERSION
        return data
    
    @staticmethod
    def _upcast_v0_9_to_v1_0(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从 v0.9（旧格式）迁移到 v1.0。
        
        迁移规则：
        - goal/conflict/outcome → intent
        - must_events → execution.units (label="action")
        - forbidden_events → constraints (type="forbidden")
        - state_delta → observables.state_changes
        """
        result = {
            "version": CONTRACT_VERSION,
            "scene_id": data.get("scene_id", f"scene_{data.get('chapter', 0)}_{data.get('scene_index', 0)}"),
            "metadata": {
                "chapter": data.get("chapter", 1),
                "scene_index": data.get("scene_index", 0),
                "arc": data.get("arc"),
            }
        }
        
        # 迁移 intent
        result["intent"] = {
            "goal": data.get("goal", ""),
            "conflict": data.get("conflict", ""),
            "expected_outcome": data.get("outcome", data.get("goal", ""))
        }
        
        # 迁移 execution.units
        units = []
        must_events = data.get("must_events", [])
        for idx, event in enumerate(must_events):
            units.append({
                "id": f"U{idx+1}",
                "label": "action",
                "description": event,
                "attributes": {}
            })
        # 如果没有 must_events，用 goal 作为默认单元
        if not units and data.get("goal"):
            units.append({
                "id": "U1",
                "label": "action",
                "description": f"完成：{data['goal']}",
                "attributes": {}
            })
        result["execution"] = {"units": units}
        
        # 迁移 constraints
        constraints = []
        forbidden = data.get("forbidden_events", [])
        for event in forbidden:
            constraints.append({
                "type": "forbidden",
                "target": event,
                "condition": None
            })
        result["constraints"] = constraints
        
        # 迁移 observable outcomes
        state_changes = []
        delta = data.get("state_delta", {})
        if delta:
            # 尝试解析 state_delta
            if "events" in delta:
                for evt in delta["events"]:
                    state_changes.append({
                        "type": "plot_flag",
                        "name": evt.get("flag", f"event_{len(state_changes)}"),
                        "value": evt.get("value", True)
                    })
            elif "characters" in delta:
                # 境界变化
                for name, info in delta["characters"].items():
                    if "realm" in info:
                        state_changes.append({
                            "type": "realm",
                            "actor": name,
                            "to_major_realm": info["realm"],
                            "to_minor_stage": info.get("level", 1)
                        })
            elif "relationships" in delta:
                for rel, val in delta["relationships"].items():
                    parts = rel.split("|")
                    if len(parts) == 2:
                        state_changes.append({
                            "type": "relationship",
                            "from_char": parts[0],
                            "to_char": parts[1],
                            "delta": val
                        })
            elif "plot_flags" in delta:
                for flag, val in delta["plot_flags"].items():
                    state_changes.append({
                        "type": "plot_flag",
                        "name": flag,
                        "value": val
                    })
        
        result["observables"] = {
            "state_changes": state_changes,
            "story_events": [],
            "narrative_flags": []
        }
        
        return result


# ============================================================================
# 工厂方法 - 从旧格式创建 Contract
# ============================================================================

# src/writing/planning_contract.py (文件末尾)

# ============================================================================
# 工厂方法 - 从旧格式创建 Contract
# ============================================================================

def create_contract_from_dict(data: Dict[str, Any]) -> PlanningContract:
    """
    从字典创建 Planning Contract（自动处理版本迁移）。
    
    Args:
        data: 原始数据（任何版本）
        
    Returns:
        PlanningContract v1.0
    """
    import logging
    logger = logging.getLogger("writing.planning_contract")
    
    # 1. 先执行版本迁移
    upcasted = ContractUpcaster.upcast(data)
    
    # 2. 修复：显式保留 scene_spec（v2.1 新增字段）
    scene_spec = data.get("scene_spec")
    if scene_spec is not None:
        # scene_spec 可能已经在 upcasted 中，也可能没有
        # 直接覆盖确保存在
        upcasted["scene_spec"] = scene_spec
        logger.info(f"✅ create_contract_from_dict: 保留 scene_spec (function={scene_spec.get('narrative_function', 'unknown')})")
    else:
        # 如果 upcasted 中有 scene_spec 但原始数据没有，保留 upcasted 的
        # 实际上 upcast 不会生成 scene_spec，所以这不会发生
        pass
    
    # 3. 构建 Contract
    try:
        contract = PlanningContract(**upcasted)
        if contract.scene_spec:
            logger.info(f"✅ PlanningContract 包含 scene_spec: {contract.scene_spec.narrative_function}")
        return contract
    except Exception as e:
        logger.error(f"❌ PlanningContract 构建失败: {e}")
        raise