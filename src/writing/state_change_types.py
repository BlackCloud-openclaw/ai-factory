# src/writing/state_change_types.py
"""
Phase 14.0A-2: StateChange 类型枚举

此文件作为 StateChangeType 的单一事实来源，
供 ContractNormalizer 和 ContractValidator 共同引用。
"""

from enum import Enum


class StateChangeType(str, Enum):
    """六类 StateChange（≤6，ADR-049.2-C2）"""
    KNOWLEDGE_GAIN = "knowledge_gain"
    INVENTORY_ACQUIRE = "inventory_acquire"
    LOCATION_CHANGE = "location_change"
    REALM_CHANGE = "realm_change"
    RELATIONSHIP_CHANGE = "relationship_change"
    PLOT_FLAG = "plot_flag"

    @classmethod
    def values(cls):
        return [item.value for item in cls]

    @classmethod
    def count(cls) -> int:
        return len(cls)