# src/writing/contracts/event_matcher.py

import logging
from typing import Dict, Any

from src.writing.events import EventType
from src.writing.planning_contract import StateChange

logger = logging.getLogger(__name__)


def _safe_number(value) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


class ContractEventMatcher:
    @staticmethod
    def match(change: StateChange, event: Dict[str, Any]) -> bool:
        event_type = event.get("type")
        if event_type == EventType.PLOT_FLAG_SET.value:
            return (event.get("flag") == change.name and
                    event.get("value") == change.value)
        elif event_type == EventType.ITEM_ACQUIRE.value:
            return (event.get("actor") == change.actor and
                    event.get("item") == change.item)
        elif event_type == EventType.LOCATION_ENTER.value:
            return (event.get("actor") == change.actor and
                    event.get("location") == change.location)
        elif event_type == EventType.REALM_UPGRADE.value:
            return (event.get("actor") == change.actor and
                    event.get("to_major_realm") == change.to_major_realm and
                    event.get("to_minor_stage") == change.to_minor_stage)
        elif event_type == EventType.RELATIONSHIP_CHANGE.value:
            event_delta = _safe_number(event.get("delta"))
            change_delta = _safe_number(change.delta)
            return (event.get("from_char") == change.from_char and
                    event.get("to_char") == change.to_char and
                    abs(event_delta - change_delta) <= 5)
        elif event_type == EventType.DISCOVERY.value:
            # Adapter: knowledge_gain → discovery
            name_match = event.get("discovery") == change.name
            if not name_match:
                return False
            # If change has actor, ensure discoverer matches; else allow any discoverer
            if getattr(change, "actor", None) is not None:
                return event.get("discoverer") == change.actor
            return True
        return False