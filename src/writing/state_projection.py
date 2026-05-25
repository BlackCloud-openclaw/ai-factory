"""状态投影层 - 将 WorldState 转换为生成时的硬约束"""

from typing import Dict, Any, List, Optional
from .world_state import WorldState, Realm
import logging

logger = logging.getLogger(__name__)

REALM_ORDER = ["炼气", "筑基", "金丹", "元婴", "化神", "炼虚", "合体", "大乘", "渡劫"]


def extract_hard_constraints(
    world_state: WorldState,
    protagonist: str = "林逸",
    max_items: int = 8,
    max_flags: int = 5,
    max_relationships: int = 3,
) -> Dict[str, Any]:
    constraints = {
        "protagonist": protagonist,
        "current_realm": {},
        "allowed_next_realms": [],
        "forbidden_realms": [],
        "current_location": "",
        "critical_items": [],
        "hp_status": "",
        "critical_flags": [],
        "critical_relationships": {},
    }
    
    # ========== 1. 主角境界（始终注入）==========
    if protagonist in world_state.characters:
        char = world_state.characters[protagonist]
        current_major = char.realm.value
        current_level = char.realm_level
        constraints["current_realm"] = {
            "major": current_major,
            "minor_level": current_level,
            "full": char.full_realm(),
        }
        
        # 计算允许的下一境界
        try:
            current_idx = REALM_ORDER.index(current_major)
            next_major = REALM_ORDER[current_idx + 1] if current_idx + 1 < len(REALM_ORDER) else None
        except ValueError:
            next_major = None
        
        if current_level < 9:
            constraints["allowed_next_realms"].append(f"{current_major}{current_level + 1}层")
        elif current_level == 9 and next_major:
            constraints["allowed_next_realms"].append(f"{next_major}一层")
        
        for r in REALM_ORDER:
            if r != current_major and r != next_major:
                constraints["forbidden_realms"].append(r)
    else:
        # 如果主角不存在，记录警告并设置默认值
        logger.warning(f"Protagonist {protagonist} not found in world_state, using defaults. Available characters: {list(world_state.characters.keys())}")
        constraints["current_realm"] = {
            "major": "炼气",
            "minor_level": 1,
            "full": "炼气一层"
        }
        constraints["allowed_next_realms"] = ["炼气二层"]
        constraints["forbidden_realms"] = ["筑基", "金丹", "元婴", "化神", "炼虚", "合体", "大乘", "渡劫"]
    
    # ========== 2. 生命值 ==========
    if protagonist in world_state.characters:
        hp = world_state.characters[protagonist].hp
        if hp <= 30:
            constraints["hp_status"] = "重伤"
        elif hp <= 70:
            constraints["hp_status"] = "轻伤"
        else:
            constraints["hp_status"] = "健康"
    
    # ========== 3. 当前位置 ==========
    if world_state.map.current:
        constraints["current_location"] = world_state.map.current
    
    # ========== 4. 关键物品（Top-K）==========
    if protagonist in world_state.characters:
        inventory = world_state.characters[protagonist].inventory
        item_priority = {
            "神秘玉佩": 100, "青锋剑": 90, "千年雷击木": 80,
            "寒铁矿": 70, "血煞丹": 60, "赤炎虎内丹": 50,
        }
        sorted_items = sorted(
            inventory,
            key=lambda x: item_priority.get(x, 0),
            reverse=True
        )
        constraints["critical_items"] = sorted_items[:max_items]
    
    # ========== 5. 关键剧情标记（Top-K）==========
    important_flags = [
        "玉佩觉醒", "封魔阵异动", "血色禁制触发",
        "师徒决裂", "金丹突破成功", "妖兽内丹现世"
    ]
    active_flags = [f for f in important_flags if world_state.global_flags.get(f)]
    constraints["critical_flags"] = active_flags[:max_flags]
    
    # ========== 6. 关键关系（Top-K）==========
    if protagonist in world_state.characters:
        char = world_state.characters[protagonist]
        sorted_rels = sorted(
            char.relationships.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        constraints["critical_relationships"] = dict(sorted_rels[:max_relationships])
    
    return constraints


def format_constraints_as_xml(constraints: Dict[str, Any]) -> str:
    """格式化为 XML + 优先级 banner"""
    lines = [
        "=" * 60,
        "⚠️ 以下约束优先级高于任何剧情张力 ⚠️",
        "如果剧情发展与状态约束冲突，必须服从状态约束。",
        "=" * 60,
        "",
        "【以下为世界状态硬约束 - 生成内容必须严格遵守】",
        "【禁止生成违反 forbidden_realms 的任何事件】",
        "【突破只能发生在 allowed_next_realms 范围内】",
        "",
        "<world_state_constraints>"
    ]
    
    lines.append(f'  <protagonist>{constraints["protagonist"]}</protagonist>')
    lines.append('  <current_realm>')
    for k, v in constraints["current_realm"].items():
        lines.append(f'    <{k}>{v}</{k}>')
    lines.append('  </current_realm>')
    
    if constraints["allowed_next_realms"]:
        lines.append('  <allowed_next_realms>')
        for r in constraints["allowed_next_realms"]:
            lines.append(f'    <realm>{r}</realm>')
        lines.append('  </allowed_next_realms>')
    else:
        lines.append('  <allowed_next_realms>（当前无法突破）</allowed_next_realms>')
    
    if constraints["forbidden_realms"]:
        lines.append('  <forbidden_realms>')
        for r in constraints["forbidden_realms"]:
            lines.append(f'    <realm>{r}</realm>')
        lines.append('  </forbidden_realms>')
    
    if constraints["hp_status"]:
        lines.append(f'  <hp_status>{constraints["hp_status"]}</hp_status>')
    
    if constraints["current_location"]:
        lines.append(f'  <current_location>{constraints["current_location"]}</current_location>')
    
    if constraints["critical_items"]:
        lines.append('  <critical_items>')
        for item in constraints["critical_items"]:
            lines.append(f'    <item>{item}</item>')
        lines.append('  </critical_items>')
    
    if constraints["critical_flags"]:
        lines.append('  <critical_flags>')
        for flag in constraints["critical_flags"]:
            lines.append(f'    <flag>{flag}</flag>')
        lines.append('  </critical_flags>')
    
    if constraints["critical_relationships"]:
        lines.append('  <critical_relationships>')
        for rel, val in constraints["critical_relationships"].items():
            lines.append(f'    <relationship target="{rel}" value="{val}"/>')
        lines.append('  </critical_relationships>')
    
    lines.append('</world_state_constraints>')
    lines.append("")
    lines.append("【重要提醒】")
    lines.append("1. 突破只能小境界逐级提升，禁止跨越大境界")
    lines.append("2. 如果伤势为重，优先疗伤而非战斗")
    lines.append("3. 仅修正违反约束的部分，不要重写整个场景")
    
    return "\n".join(lines)


def get_state_diff_message(error_details: Dict[str, Any]) -> str:
    """生成精确的状态差异消息"""
    error_type = error_details.get("type")
    if error_type == "realm_upgrade_violation":
        return f"""❌ 境界约束违反：

- 当前境界：{error_details.get('current_realm', '未知')}
- 允许突破：{error_details.get('expected_realm', '未知')}
- 实际生成：{error_details.get('actual_realm', '未知')}

请仅修正境界部分，不要重写整个场景。"""
    
    return error_details.get("message", "请检查生成内容是否违反约束")