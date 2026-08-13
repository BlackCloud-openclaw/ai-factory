# src/writing/contracts/event_mapping.py

from enum import Enum
from typing import List, Union
import logging

from src.writing.events import EventType

logger = logging.getLogger(__name__)

# 规划语义 → Runtime 事件类型列表（支持一对多）
CONTRACT_TO_RUNTIME = {
    "plot_flag": [EventType.PLOT_FLAG_SET],
    "inventory_acquire": [EventType.ITEM_ACQUIRE],
    "location_change": [EventType.LOCATION_ENTER],
    "realm_change": [EventType.REALM_UPGRADE],
    "relationship_change": [EventType.RELATIONSHIP_CHANGE],
    # Adapter: knowledge_gain currently maps to discovery
    "knowledge_gain": [EventType.DISCOVERY],
}

# 运行事件 → 契约类型的默认推断（非严格逆映射，仅作提示）
RUNTIME_TO_CONTRACT_HINT = {
    EventType.PLOT_FLAG_SET: "plot_flag",
    EventType.ITEM_ACQUIRE: "inventory_acquire",
    EventType.LOCATION_ENTER: "location_change",
    EventType.REALM_UPGRADE: "realm_change",
    EventType.RELATIONSHIP_CHANGE: "relationship_change",
    EventType.DISCOVERY: "knowledge_gain",
}


class ContractEventResolver:
    @staticmethod
    def resolve(state_change_type: Union[str, Enum]) -> List[EventType]:
        if hasattr(state_change_type, "value"):
            state_change_type = state_change_type.value

        mapped = CONTRACT_TO_RUNTIME.get(state_change_type)
        if mapped is not None:
            if state_change_type == "knowledge_gain":
                logger.info(
                    "KNOWLEDGE_GAIN_EVENT_ADAPTER_USED: knowledge_gain -> discovery"
                )
            return mapped

        try:
            return [EventType(state_change_type)]
        except ValueError:
            logger.warning(
                "UNKNOWN_CONTRACT_EVENT_TYPE: %s (will be treated as unmatched)",
                state_change_type
            )
            return []