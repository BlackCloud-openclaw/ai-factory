#!/usr/bin/env python3
"""
测试 ContractUpcaster - 独立版本（不依赖 src/writing 包）
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal
from datetime import datetime
from pprint import pprint

# 只依赖 Pydantic
from pydantic import BaseModel, Field, field_validator

# ============================================================================
# Planning Contract 模型（直接从 planning_contract.py 复制）
# ============================================================================

class ExecutionUnit(BaseModel):
    id: str
    label: Literal["action", "beat", "intent", "constraint"]
    description: str
    attributes: Dict[str, Any] = Field(default_factory=dict)


class Execution(BaseModel):
    units: List[ExecutionUnit] = Field(default_factory=list)


class Constraint(BaseModel):
    type: Literal["required", "forbidden", "before", "after", "exclusive", "at_least_once"]
    target: str
    condition: Optional[str] = None
    refs: Optional[List[str]] = None

    @field_validator("target")
    @classmethod
    def validate_target(cls, v: str) -> str:
        if not v or len(v.strip()) < 2:
            raise ValueError("约束目标至少需要2个字符")
        return v.strip()


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


class StoryEvent(BaseModel):
    type: Literal["dialogue", "discovery", "combat", "decision"]
    description: str
    participants: List[str] = Field(default_factory=list)
    importance: Literal["low", "normal", "high", "critical"] = Field(default="normal")


class NarrativeFlag(BaseModel):
    name: str
    value: Any


class Observables(BaseModel):
    state_changes: List[StateChange] = Field(default_factory=list)
    story_events: List[StoryEvent] = Field(default_factory=list)
    narrative_flags: List[NarrativeFlag] = Field(default_factory=list)


class Intent(BaseModel):
    goal: str
    conflict: str
    expected_outcome: str

    @field_validator("goal", "conflict", "expected_outcome")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or len(v.strip()) < 3:
            raise ValueError("字段至少需要3个字符")
        return v.strip()


class ContractMetadata(BaseModel):
    chapter: int
    scene_index: int
    arc: Optional[str] = None
    created_at: Optional[datetime] = Field(default_factory=datetime.now)

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


class PlanningContract(BaseModel):
    version: str = Field(default="1.0")
    scene_id: str
    intent: Intent
    execution: Execution = Field(default_factory=Execution)
    observables: Observables = Field(default_factory=Observables)
    constraints: List[Constraint] = Field(default_factory=list)
    metadata: ContractMetadata


# ============================================================================
# ContractUpcaster（直接从 planning_contract.py 复制）
# ============================================================================

CONTRACT_VERSION = "1.0"


class ContractUpcaster:
    """Contract 版本迁移器"""
    
    @staticmethod
    def upcast(data: Dict[str, Any]) -> Dict[str, Any]:
        version = data.get("version", "0.9")
        
        if version == "0.9":
            data = ContractUpcaster._upcast_v0_9_to_v1_0(data)
        
        data["version"] = CONTRACT_VERSION
        return data
    
    @staticmethod
    def _upcast_v0_9_to_v1_0(data: Dict[str, Any]) -> Dict[str, Any]:
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
                        "type": "plot_flag",
                        "name": evt.get("flag", f"event_{len(state_changes)}"),
                        "value": evt.get("value", True)
                    })
            elif "characters" in delta:
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


def create_contract_from_dict(data: Dict[str, Any]) -> PlanningContract:
    """从字典创建 Contract"""
    upcasted = ContractUpcaster.upcast(data)
    return PlanningContract(**upcasted)


# ============================================================================
# 测试函数
# ============================================================================

def test_upcaster():
    """测试 Upcaster 的迁移功能"""
    
    # 旧格式数据（v0.9）
    old_data = {
        "version": "0.9",
        "scene_id": "legacy_scene_001",
        "chapter": 3,
        "scene_index": 1,
        "goal": "林逸尝试说服苏清雪合作探索秘境",
        "conflict": "苏清雪对林逸的能力存疑，且有自己的秘密",
        "outcome": "苏清雪勉强同意合作，但保留关键信息",
        "must_events": [
            "林逸在藏书阁找到苏清雪",
            "林逸抛出合作意向",
            "苏清雪冷淡回应",
            "林逸透露青铜钥匙碎片",
            "苏清雪提出'信息对等'条件",
            "林逸同意"
        ],
        "forbidden_events": [
            "苏清雪直接答应合作",
            "林逸展示全部底牌"
        ],
        "state_delta": {
            "relationships": {
                "林逸|苏清雪": 15
            },
            "plot_flags": {
                "cooperation_started": True
            }
        }
    }
    
    print("=" * 80)
    print("测试 ContractUpcaster：旧格式 → v1.0")
    print("=" * 80)
    print("\n📄 旧数据（v0.9）:")
    pprint(old_data)
    
    # 迁移
    print("\n⬆️ 执行迁移...")
    migrated = ContractUpcaster.upcast(old_data)
    
    print("\n✅ 迁移结果（v1.0）:")
    pprint(migrated)
    
    # 验证迁移后的数据能否通过 Schema 验证
    print("\n🔍 验证迁移后的数据是否符合 PlanningContract v1.0 Schema...")
    try:
        contract = create_contract_from_dict(old_data)
        print("✅ 迁移后的数据通过 Schema 验证")
        print(f"   - scene_id: {contract.scene_id}")
        print(f"   - intent.goal: {contract.intent.goal[:30]}...")
        print(f"   - execution.units 数量: {len(contract.execution.units)}")
        print(f"   - constraints 数量: {len(contract.constraints)}")
        print(f"   - observables.state_changes 数量: {len(contract.observables.state_changes)}")
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False
    
    return True


def test_empty_migration():
    """测试空数据迁移"""
    print("\n" + "=" * 80)
    print("测试空数据迁移")
    print("=" * 80)
    
    empty_data = {
        "goal": "测试目标",
        "conflict": "测试冲突"
    }
    
    result = ContractUpcaster.upcast(empty_data)
    
    if "intent" in result:
        print("✅ 空数据迁移成功")
        print(f"   - intent.goal: {result['intent']['goal']}")
        print(f"   - execution.units: {len(result['execution']['units'])}")
        return True
    else:
        print("❌ 空数据迁移失败")
        return False


def main():
    """运行所有测试"""
    print("\n🚀 启动 ContractUpcaster 测试套件（独立版）\n")
    
    test_result = test_upcaster()
    empty_result = test_empty_migration()
    
    if test_result and empty_result:
        print("\n" + "=" * 80)
        print("✅ 所有测试通过")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ 测试失败")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())