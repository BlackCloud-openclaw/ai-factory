#!/usr/bin/env python3
"""
Contract Schema 验证脚本（独立版）
不依赖 src/writing 包，避免循环导入
直接定义 PlanningContract 的 Pydantic 模型
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Literal
from datetime import datetime

# 只依赖 Pydantic，不依赖项目其他模块
from pydantic import BaseModel, Field, field_validator

# 添加项目根目录到路径（仅用于查找 YAML 文件）
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import yaml


# ============================================================================
# Planning Contract 模型（直接从 planning_contract.py 复制）
# ============================================================================

class ExecutionUnit(BaseModel):
    id: str = Field(..., description="单元唯一标识")
    label: Literal["action", "beat", "intent", "constraint"] = Field(
        ..., description="单元类型"
    )
    description: str = Field(..., description="自然语言描述")
    attributes: Dict[str, Any] = Field(default_factory=dict)


class Execution(BaseModel):
    units: List[ExecutionUnit] = Field(default_factory=list)


class Constraint(BaseModel):
    type: Literal["required", "forbidden", "before", "after", "exclusive", "at_least_once"] = Field(
        ..., description="约束类型"
    )
    target: str = Field(..., description="约束目标")
    condition: Optional[str] = Field(default=None)
    refs: Optional[List[str]] = Field(default=None)

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
    scene_id: str = Field(..., description="场景唯一标识")
    intent: Intent
    execution: Execution = Field(default_factory=Execution)
    observables: Observables = Field(default_factory=Observables)
    constraints: List[Constraint] = Field(default_factory=list)
    metadata: ContractMetadata


# ============================================================================
# 验证函数
# ============================================================================

def find_all_yaml_files(base_dir: Path) -> List[Path]:
    return list(base_dir.rglob("*.yaml")) + list(base_dir.rglob("*.yml"))


def load_yaml_file(filepath: Path) -> Dict[str, Any]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def validate_contract(data: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        if "planning_contract" in data:
            contract_data = data["planning_contract"]
        else:
            contract_data = data
        
        contract = PlanningContract(**contract_data)
        return True, f"✅ Contract 有效 (scene_id={contract.scene_id})"
    except Exception as e:
        return False, f"❌ {e}"


def generate_summary_report(
    results: List[Tuple[Path, bool, str]]
) -> str:
    lines = [
        "=" * 80,
        "Planning Contract Schema 验证报告",
        "=" * 80,
        "",
        f"总文件数: {len(results)}",
        "",
    ]
    
    valid_count = sum(1 for _, valid, _ in results if valid)
    invalid_files = [(p, msg) for p, valid, msg in results if not valid]
    
    for filepath, valid, msg in results:
        status = "✅" if valid else "❌"
        rel_path = filepath.relative_to(filepath.parent.parent)
        lines.append(f"{status} {rel_path}")
        if not valid:
            lines.append(f"   {msg}")
    
    lines.append("")
    lines.append("-" * 80)
    lines.append(f"总计: {valid_count}/{len(results)} 个文件通过验证")
    
    if invalid_files:
        lines.append("")
        lines.append("=" * 80)
        lines.append("失败详情")
        lines.append("=" * 80)
        for filepath, msg in invalid_files:
            lines.append(f"\n📄 {filepath.relative_to(filepath.parent.parent)}")
            lines.append(f"  {msg}")
    
    return "\n".join(lines)


def main():
    base_dir = Path(__file__).parent / "contract_examples"
    
    if not base_dir.exists():
        print(f"❌ 目录不存在: {base_dir}")
        print("请先创建 contract_examples 目录并放入示例文件")
        return
    
    yaml_files = find_all_yaml_files(base_dir)
    
    if not yaml_files:
        print(f"❌ 在 {base_dir} 中没有找到 YAML 文件")
        return
    
    print(f"找到 {len(yaml_files)} 个 YAML 文件\n")
    
    results = []
    for filepath in yaml_files:
        rel_path = filepath.relative_to(base_dir)
        print(f"验证: {rel_path}")
        try:
            data = load_yaml_file(filepath)
            valid, msg = validate_contract(data)
            results.append((filepath, valid, msg))
            if valid:
                print("  ✅ 通过\n")
            else:
                print(f"  ❌ 失败\n")
        except Exception as e:
            results.append((filepath, False, f"加载失败: {e}"))
            print(f"  ❌ 加载失败: {e}\n")
    
    report = generate_summary_report(results)
    print(report)
    
    # 保存报告
    report_dir = base_dir / "reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / "validation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n📄 报告已保存至: {report_path}")


if __name__ == "__main__":
    main()