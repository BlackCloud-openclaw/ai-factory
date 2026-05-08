# src/writing/invariants.py
from typing import Dict, Any, Tuple
from .events import Event, EVENT_CHARACTER_UPDATE, EVENT_TIMELINE_ADD

# 默认境界顺序（可被大纲覆盖）
DEFAULT_REALM_ORDER = ["炼气", "筑基", "金丹", "元婴", "化神"]

def validate_event(state: Dict[str, Any], event: Event, realm_order: list = None) -> Tuple[bool, str]:
    """基础不变性检查（可配置境界顺序）"""
    if event.type == EVENT_CHARACTER_UPDATE:
        char_name = event.payload["name"]
        updates = event.payload.get("updates", {})
        if "realm" in updates:
            order = realm_order or DEFAULT_REALM_ORDER
            old_realm = state.get("characters", {}).get(char_name, {}).get("realm")
            new_realm = updates["realm"]
            if old_realm and new_realm:
                try:
                    old_idx = order.index(old_realm)
                    new_idx = order.index(new_realm)
                except ValueError:
                    return False, f"境界 '{old_realm}' 或 '{new_realm}' 不在预定义顺序中"
                if new_idx - old_idx > 1:
                    return False, f"境界不能跨级提升（{old_realm} → {new_realm}）"
        # 其他角色更新规则可添加...
    elif event.type == EVENT_TIMELINE_ADD:
        # 示例：同一章内不能重复同一事件（根据需求可配置）
        pass
    return True, ""