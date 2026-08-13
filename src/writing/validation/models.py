from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Optional, Mapping, Any

class ContractSeverity(Enum):
    BLOCKING = "blocking"   # 必须实现，否则重试
    WARNING = "warning"     # 可选，记录但不阻断

@dataclass(frozen=True)
class MissingContractChange:
    """Validator 输出的缺失契约信息（投影），不依赖 PlanningContract 内部模型"""
    type: str                           # 契约类型，如 "relationship_change"
    description: str                    # 面向自然语言的描述
    severity: ContractSeverity = ContractSeverity.BLOCKING
    actor: Optional[str] = None
    fields: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    source: str = "planning_contract"   # 来源标识
    contract_id: Optional[str] = None
    confidence: float = 1.0

    def __post_init__(self):
        # 确保 fields 为 MappingProxyType（不可变）
        if not isinstance(self.fields, MappingProxyType):
            object.__setattr__(self, 'fields', MappingProxyType(self.fields))
            
    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "description": self.description,
            "severity": self.severity.value,
            "actor": self.actor,
            "fields": dict(self.fields),
            "source": self.source,
            "contract_id": self.contract_id,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MissingContractChange":
        data = dict(data)

        severity = data.get("severity")
        if isinstance(severity, str):
            try:
                data["severity"] = ContractSeverity(severity.lower())
            except ValueError:
                data["severity"] = ContractSeverity.BLOCKING

        if data.get("fields") is None:
            data["fields"] = {}

        return cls(**data)