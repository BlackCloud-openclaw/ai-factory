# src/writing/reducer.py
from typing import Dict, Any
from .events import Event, EVENT_CHARACTER_UPDATE, EVENT_TIMELINE_ADD, EVENT_WORLD_RULE_ADD

def apply_event(state: Dict[str, Any], event: Event) -> Dict[str, Any]:
    """增量更新状态副本（纯函数）"""
    new_state = state.copy()
    if event.type == EVENT_CHARACTER_UPDATE:
        char_name = event.payload["name"]
        # 确保 characters 子字典存在
        if "characters" not in new_state:
            new_state["characters"] = {}
        if char_name not in new_state["characters"]:
            new_state["characters"][char_name] = {}
        # 合并更新
        new_state["characters"][char_name].update(event.payload["updates"])
    elif event.type == EVENT_TIMELINE_ADD:
        if "timeline" not in new_state:
            new_state["timeline"] = []
        new_state["timeline"].append(event.payload)
    elif event.type == EVENT_WORLD_RULE_ADD:
        if "world_rules" not in new_state:
            new_state["world_rules"] = []
        new_state["world_rules"].append(event.payload["rule"])
    # 其他事件类型...
    return new_state

def reduce_state(events: list[Event]) -> Dict[str, Any]:
    """从事件列表全量构建状态（用于初始化）"""
    state = {"characters": {}, "timeline": [], "world_rules": []}
    for event in sorted(events, key=lambda e: e.sequence_id):
        state = apply_event(state, event)
    return state