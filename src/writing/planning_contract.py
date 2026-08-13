# src/writing/planning_contract.py
"""
Planning Contract v1.0 - AI Factory 稳定规划接口 (Stable Planning Interface)

Planning Contract 是 AI Factory v2.0 的核心抽象，定义了 Planner 对每个场景的承诺（Promise）。
它是一个与 Runtime 无关的领域 DSL，所有 Agent（Planner、Writer、Validator）都围绕此契约工作。

核心设计原则：
- Contract 是冻结的接口，允许 Planner/Writer/Validator/Runtime 独立演进
- 不包含任何 Runtime 概念（Event、Projection、Operation）
- 不包含任何实验变量（representation、density、temperature）
- 从第一天支持版本迁移（ContractUpcaster）

Phase 13.2.3A 扩展：
- 增加 SignalSource 枚举（信号来源追踪）
- 增加 ContractEnrichment 元数据
- StateChange 增加 id 和 source 字段
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from datetime import datetime
import hashlib
import json
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# 版本定义
# ============================================================================

CONTRACT_VERSION = "1.0"


# ============================================================================
# Phase 13.2.3A: Signal Source & Enrichment Metadata
# ============================================================================

class SignalSource(str, Enum):
    """信号来源枚举 - 确保 Contract 信号可追踪。"""
    UNKNOWN = "unknown"        # 未声明来源（防御性默认）
    LLM = "llm"                # Planner 直接生成
    INFERRED = "inferred"      # 系统从 must_events / context 推断
    SYSTEM = "system"          # 系统默认值（兜底）
    NORMALIZED = "normalized"  # 从已有信号标准化派生


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
    id: Optional[str] = Field(
        default=None,
        description="约束唯一标识（推断时自动生成）"
    )
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

    Phase 13.2.3A 扩展：
    - 增加 id 字段（稳定标识）
    - 增加 source 字段（信号来源追踪）

    Phase 14.0A-2 扩展：
    - 增加 confidence 字段（推断置信度）
    """

    id: str = Field(
        default="",
        description="稳定标识，由系统生成或推断"
    )

    type: str = Field(
        ...,
        description="状态变化类型，必须属于 StateChangeType 枚举值"
    )

    source: SignalSource = Field(
        default=SignalSource.UNKNOWN,
        description="信号来源（llm / inferred / system / unknown）"
    )

    confidence: float = Field(
        default=0.0,
        description="推断置信度，范围 0.0-1.0，仅当 source=INFERRED 时有效",
        ge=0.0,
        le=1.0,
    )

    # ----- 以下字段根据 type 不同按需使用 -----

    # Plot Flag 相关
    name: Optional[str] = Field(
        None,
        description="标记名称（plot_flag 类型使用）"
    )
    value: Optional[Any] = Field(
        None,
        description="标记值（plot_flag 类型使用）"
    )

    # Relationship 相关
    from_char: Optional[str] = Field(
        None,
        description="关系发起方（relationship 类型使用）"
    )
    to_char: Optional[str] = Field(
        None,
        description="关系接收方（relationship 类型使用）"
    )
    delta: Optional[int] = Field(
        None,
        description="关系变化量（relationship 类型使用）"
    )

    # Inventory 相关
    actor: Optional[str] = Field(
        None,
        description="操作者（inventory/realm/location/hp 类型使用）"
    )
    item: Optional[str] = Field(
        None,
        description="物品名称（inventory 类型使用）"
    )
    operation: Optional[Literal["acquire", "lose"]] = Field(
        None,
        description="操作类型（inventory 类型使用）"
    )
    quantity: Optional[int] = Field(
        1,
        description="数量（inventory 类型使用）"
    )

    # Realm 相关
    to_major_realm: Optional[str] = Field(
        None,
        description="目标大境界（realm 类型使用）"
    )
    to_minor_stage: Optional[int] = Field(
        None,
        description="目标小层级（realm 类型使用）"
    )

    # Location 相关
    location: Optional[str] = Field(
        None,
        description="目标地点（location 类型使用）"
    )

    # HP 相关
    new_hp: Optional[int] = Field(
        None,
        description="新 HP 值（hp 类型使用）"
    )

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """序列化时清理空值字段"""
        data = super().model_dump(**kwargs)
        # 保留必要的核心字段
        cleaned = {
            "id": data.get("id", ""),
            "type": data["type"],
            "source": data.get("source", SignalSource.UNKNOWN),
            "confidence": data.get("confidence", 0.0),
        }

        # 根据 type 选择性添加其他字段
        if self.type == "plot_flag":
            if data.get("name") is not None:
                cleaned["name"] = data["name"]
            if data.get("value") is not None:
                cleaned["value"] = data["value"]
        elif self.type == "relationship":
            if data.get("from_char") is not None:
                cleaned["from_char"] = data["from_char"]
            if data.get("to_char") is not None:
                cleaned["to_char"] = data["to_char"]
            if data.get("delta") is not None:
                cleaned["delta"] = data["delta"]
        elif self.type in ["inventory", "realm", "location", "hp"]:
            # 添加所有非空字段
            for key in ["actor", "item", "operation", "quantity", 
                        "to_major_realm", "to_minor_stage", "location", "new_hp"]:
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
# ContractEnrichment - 标准化元数据 (Phase 13.2.3A)
# ============================================================================

class ContractEnrichment(BaseModel):
    """标准化元数据 - 记录所有补全和推断操作。"""
    version: str = Field(default="1.0", description="标准化器版本")
    enriched: bool = Field(default=False, description="是否执行了补全操作")
    sources: Dict[str, SignalSource] = Field(
        default_factory=dict,
        description="StateChange ID -> 来源映射"
    )
    rules_applied: List[str] = Field(
        default_factory=list,
        description="触发的推断规则列表"
    )
    normalized_at: Optional[datetime] = Field(
        default=None,
        description="标准化时间戳"
    )
    normalizer_version: str = Field(
        default="13.2.3A-v1.2",
        description="Normalizer 实现版本"
    )
    input_hash: str = Field(
        default="",
        description="输入内容的哈希，用于幂等判断"
    )

    def mark_enriched(self, input_hash: str):
        self.enriched = True
        self.normalized_at = datetime.now()
        self.input_hash = input_hash


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
        scene_spec: v2.1 场景规格（可选）
        enrichment: Phase 13.2.3A 标准化元数据
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
    # Phase 13.2.3A 新增
    enrichment: ContractEnrichment = Field(
        default_factory=ContractEnrichment,
        description="标准化元数据（来源追踪）"
    )

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        # 强制保留 observables，即使为空列表
        kwargs.setdefault("exclude_unset", False)
        kwargs.setdefault("exclude_defaults", False)
        data = super().model_dump(**kwargs)
        # 确保 observables 始终存在（防御性）
        if "observables" not in data or not data["observables"]:
            data["observables"] = {
                "state_changes": [],
                "story_events": [],
                "narrative_flags": [],
            }
        return data


# ============================================================================
# ContractUpcaster - 版本迁移
# ============================================================================

class ContractUpcaster:
    @staticmethod
    def upcast(data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("🔥🔥🔥 UPCATER EXECUTING 🔥🔥🔥")
        logger.info(f"🔍 Upcaster input observables: {data.get('observables', 'MISSING')}")
        version = data.get("version", "0.9")
        
        if version == "0.9":
            data = ContractUpcaster._upcast_v0_9_to_v1_0(data)
        
        # v2.1: 保留 scene_spec
        if "scene_spec" in data and data["scene_spec"] is not None:
            if "scene_spec" not in data:
                data["scene_spec"] = None
            if isinstance(data["scene_spec"], dict):
                data["scene_spec"] = data["scene_spec"]
        
        # Phase 13.2.3A: 补充 enrichment 字段
        if "enrichment" not in data or data["enrichment"] is None:
            data["enrichment"] = ContractEnrichment().model_dump()
        
        data["version"] = CONTRACT_VERSION
        logger.info(f"🔍 Upcaster output observables: {data.get('observables', 'MISSING')}")
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
            if "events" in delta:
                for evt in delta["events"]:
                    state_changes.append({
                        "id": f"migrated_{len(state_changes)}",
                        "type": "plot_flag",
                        "source": "unknown",
                        "name": evt.get("flag", f"event_{len(state_changes)}"),
                        "value": evt.get("value", True)
                    })
            elif "characters" in delta:
                for name, info in delta["characters"].items():
                    if "realm" in info:
                        state_changes.append({
                            "id": f"migrated_{len(state_changes)}",
                            "type": "realm",
                            "source": "unknown",
                            "actor": name,
                            "to_major_realm": info["realm"],
                            "to_minor_stage": info.get("level", 1)
                        })
            elif "relationships" in delta:
                for rel, val in delta["relationships"].items():
                    parts = rel.split("|")
                    if len(parts) == 2:
                        state_changes.append({
                            "id": f"migrated_{len(state_changes)}",
                            "type": "relationship",
                            "source": "unknown",
                            "from_char": parts[0],
                            "to_char": parts[1],
                            "delta": val
                        })
            elif "plot_flags" in delta:
                for flag, val in delta["plot_flags"].items():
                    state_changes.append({
                        "id": f"migrated_{len(state_changes)}",
                        "type": "plot_flag",
                        "source": "unknown",
                        "name": flag,
                        "value": val
                    })
        
        result["observables"] = {
            "state_changes": state_changes,
            "story_events": [],
            "narrative_flags": []
        }
        
        # Phase 13.2.3A: 添加 enrichment
        result["enrichment"] = ContractEnrichment().model_dump()
        
        return result


# ============================================================================
# 工厂方法 - 从旧格式创建 Contract
# ============================================================================

def create_contract_from_dict(data: Dict[str, Any]) -> PlanningContract:
    logger.info("🔥🔥🔥 CONTRACT BUILDER EXECUTING 🔥🔥🔥")
    logger.info(f"🔍 Input data observables: {data.get('observables', 'MISSING')}")    
    # 1. 先执行版本迁移
    upcasted = ContractUpcaster.upcast(data)
    
    # 2. 保留 scene_spec（v2.1 新增字段）
    scene_spec = data.get("scene_spec")
    if scene_spec is not None:
        upcasted["scene_spec"] = scene_spec
        logger.info(f"✅ create_contract_from_dict: 保留 scene_spec (function={scene_spec.get('narrative_function', 'unknown')})")
    
    # 3. 确保 enrichment 存在
    if "enrichment" not in upcasted or upcasted["enrichment"] is None:
        upcasted["enrichment"] = ContractEnrichment().model_dump()
    
    # 4. 构建 Contract
    try:
        contract = PlanningContract(**upcasted)
        if contract.scene_spec:
            logger.info(f"✅ PlanningContract 包含 scene_spec: {contract.scene_spec.narrative_function}")
        
        # ========== 🔥 插入 diff 日志 ==========
        original_observables = data.get("observables", {})
        contract_observables = contract.observables.model_dump()
        
        if original_observables.get("state_changes") and not contract_observables.get("state_changes"):
            logger.error(f"❌ CRITICAL: Contract builder dropped observables!")
            logger.error(f"   Original observables: {original_observables}")
            logger.error(f"   Contract observables:  {contract_observables}")
            # 临时修复：从原始数据恢复（诊断用，最终应修复 upcaster）
            # 但为诊断，我们先抛出异常？不，我们只记录，让测试失败。
        else:
            logger.info(f"✅ Observables preserved: {len(contract.observables.state_changes)} state_changes")
        # ==========================================
        logger.info(f"🔍 Contract observables after building: {contract.observables.model_dump()}")
        return contract
    except Exception as e:
        logger.error(f"❌ PlanningContract 构建失败: {e}")
        raise