# src/writing/narrative_entropy.py
"""
叙事熵系统 - 支持多尺度熵计算与调控

包含：
- 单值熵（兼容旧版）
- 多尺度熵报告（局部/弧线/文明）
- 熵控制器（生成调控动作）
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from src.config_loader import get_xianxia_config


# ============================================================================
# 旧版兼容数据结构（保留）
# ============================================================================

@dataclass
class NarrativeEntropyScore:
    """单值叙事熵（保留用于向后兼容）"""
    value: float = 0.0
    unresolved_arcs: int = 0
    character_overlap: float = 0.0
    consecutive_escalation: int = 0
    new_lore_rate: float = 0.0
    unresolved_relationships: int = 0


# ============================================================================
# 多尺度熵系统（新增）
# ============================================================================

class EntropyLevel(str, Enum):
    """熵的层级"""
    LOCAL = "local"           # 场景级：张力密度、信息暴露、情感消耗
    ARC = "arc"               # 弧线级：未解决弧线、连续 escalation
    CIVILIZATION = "civilization"  # 世界级：设定膨胀、战力崩坏、势力复杂度


@dataclass
class EntropyReport:
    """多尺度熵报告"""
    local: float = 0.0          # 局部熵（0~1）
    arc: float = 0.0            # 弧线熵（0~1）
    civilization: float = 0.0   # 文明熵（0~1）
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "local": self.local,
            "arc": self.arc,
            "civilization": self.civilization,
            "details": self.details,
        }


# ============================================================================
# 熵控制器（生成调控动作）
# ============================================================================

@dataclass
class ControlAction:
    """熵调控动作"""
    type: str                     # limit_scene_role, resolve_arcs, forbid_new_lore, force_low_stakes
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "params": self.params}


class EntropyController:
    """叙事稳态控制器 - 根据熵报告生成规划约束"""

    @classmethod
    def _get_thresholds(cls):
        from src.config_loader import get_xianxia_config
        cfg = get_xianxia_config()
        thresholds = cfg.entropy.get("thresholds", {})
        # 默认值兜底
        return {
            "local_warning": thresholds.get("local_warning", 0.6),
            "local_critical": thresholds.get("local_critical", 0.8),
            "arc_warning": thresholds.get("arc_warning", 0.6),
            "arc_critical": thresholds.get("arc_critical", 0.8),
            "civ_warning": thresholds.get("civ_warning", 0.7),
            "civ_critical": thresholds.get("civ_critical", 0.9),
            "escalation_warning": thresholds.get("consecutive_escalation_warning", 3),
            "escalation_critical": thresholds.get("consecutive_escalation_critical", 4),
            "reveal_density_warning": thresholds.get("reveal_density_warning", 0.5),
            "reveal_density_critical": thresholds.get("reveal_density_critical", 0.7),
        }

    @classmethod
    def regulate(cls, entropy: EntropyReport) -> List[ControlAction]:
        thresholds = cls._get_thresholds()
        actions = []
        details = entropy.details

        consecutive_esc = details.get("consecutive_escalation", 0)
        if consecutive_esc >= thresholds["escalation_critical"]:
            actions.append(ControlAction(
                type="limit_scene_role",
                params={
                    "allowed": ["AFTERMATH", "RELEASE", "TRANSITION"],
                    "forbidden": ["ESCALATION", "REVEAL", "SETUP"],
                    "reason": f"连续 {consecutive_esc} 次 escalation，必须冷却"
                }
            ))
        elif consecutive_esc >= thresholds["escalation_warning"]:
            actions.append(ControlAction(
                type="limit_scene_role",
                params={
                    "allowed": ["AFTERMATH", "RELEASE", "TRANSITION", "SETUP"],
                    "forbidden": ["ESCALATION"],
                    "reason": f"连续 {consecutive_esc} 次 escalation，禁止继续升级"
                }
            ))

        reveal_density = details.get("reveal_density", 0.0)
        if reveal_density >= thresholds["reveal_density_critical"]:
            actions.append(ControlAction(
                type="limit_scene_role",
                params={
                    "allowed": ["TRANSITION", "AFTERMATH"],
                    "forbidden": ["REVEAL", "ESCALATION"],
                    "reason": f"揭示密度过高 ({reveal_density:.0%})，必须冷却"
                }
            ))
        elif reveal_density >= thresholds["reveal_density_warning"]:
            actions.append(ControlAction(
                type="limit_scene_role",
                params={
                    "allowed": ["SETUP", "TRANSITION", "RELEASE"],
                    "forbidden": ["REVEAL"],
                    "reason": f"揭示密度过高 ({reveal_density:.0%})，暂缓新揭示"
                }
            ))

        has_aftermath = details.get("has_aftermath_recent", False)
        if not has_aftermath and consecutive_esc >= 2:
            actions.append(ControlAction(
                type="suggest_scene_role",
                params={"recommended": "AFTERMATH", "reason": "最近无余波场景，建议补充"}
            ))

        if entropy.local >= thresholds["local_critical"]:
            actions.append(ControlAction(
                type="limit_scene_role",
                params={
                    "allowed": ["AFTERMATH", "TRANSITION"],
                    "forbidden": ["ESCALATION", "REVEAL", "SETUP"]
                }
            ))
            actions.append(ControlAction(
                type="force_low_stakes",
                params={"reason": "local entropy critical"}
            ))
        elif entropy.local >= thresholds["local_warning"]:
            actions.append(ControlAction(
                type="limit_scene_role",
                params={
                    "allowed": ["AFTERMATH", "TRANSITION", "RELEASE"],
                    "forbidden": ["ESCALATION"]
                }
            ))

        if entropy.arc >= thresholds["arc_critical"]:
            actions.append(ControlAction(
                type="resolve_arcs",
                params={"max_open": 3, "force_resolve": True}
            ))
            actions.append(ControlAction(
                type="forbid_new_arcs",
                params={"duration_chapters": 2}
            ))
        elif entropy.arc >= thresholds["arc_warning"]:
            actions.append(ControlAction(
                type="resolve_arcs",
                params={"max_open": 5, "force_resolve": False}
            ))

        if entropy.civilization >= thresholds["civ_critical"]:
            actions.append(ControlAction(
                type="forbid_new_lore",
                params={"duration_chapters": 3, "types": ["character", "location", "item", "realm"]}
            ))
            actions.append(ControlAction(
                type="force_low_stakes",
                params={"reason": "civilization entropy critical"}
            ))
        elif entropy.civilization >= thresholds["civ_warning"]:
            actions.append(ControlAction(
                type="forbid_new_lore",
                params={"duration_chapters": 1, "types": ["character", "location"]}
            ))

        return actions


# ============================================================================
# 熵计算器（支持旧版与多尺度）
# ============================================================================

class NarrativeEntropyCalculator:
    """叙事熵计算器 - 同时提供单值熵和多尺度熵"""

    # 旧版权重（保留）
    WEIGHTS = {
        "unresolved_arcs": 0.25,
        "character_overlap": 0.20,
        "consecutive_escalation": 0.20,
        "new_lore_rate": 0.20,
        "unresolved_relationships": 0.15,
    }

    # ==================== 多尺度熵计算（新增） ====================

    @classmethod
    def calculate_full(
        cls,
        world_state: Any,                     # WorldState 实例
        compressed_state: Any,                # CompressedState 实例（可选）
        recent_scene_roles: List[str],        # 最近 N 个场景的角色标签列表
        recent_events: List[Dict],            # 最近事件列表（用于新 lore 率）
        active_arcs: Optional[Dict[str, Any]] = None,  # 弧线状态字典
    ) -> EntropyReport:
        """
        计算多尺度熵报告

        Args:
            world_state: 当前世界状态
            compressed_state: 压缩状态（含 arc 信息等）
            recent_scene_roles: 最近场景的 scene_role 列表
            recent_events: 最近事件列表（用于统计新 lore 引入）
            active_arcs: 活跃弧线字典（若为 None，则从 compressed_state 读取）
        """
        details = {}

        # ----- 局部熵（local）-----
        # 安全初始化所有局部变量
        consecutive_escalation = 0
        reveal_density = 0.0
        has_aftermath = False

        if not recent_scene_roles:
            local = 0.1
        else:
            # 1. 连续 escalation 比例
            for role in reversed(recent_scene_roles):
                if role == "ESCALATION":
                    consecutive_escalation += 1
                else:
                    break
            local_escalation_norm = min(consecutive_escalation / 5.0, 1.0)

            # 2. 信息暴露密度（最近场景中 REVEAL 的比例）
            reveal_count = sum(1 for role in recent_scene_roles if role == "REVEAL")
            reveal_density = min(reveal_count / max(len(recent_scene_roles), 1), 1.0)

            # 3. AFTERMATH 缺失惩罚
            has_aftermath = any(role == "AFTERMATH" for role in recent_scene_roles[-3:])
            aftermath_penalty = 0.0 if has_aftermath else 0.3

            local = min(0.4 * local_escalation_norm + 0.4 * reveal_density + 0.2 * aftermath_penalty, 1.0)

        details["consecutive_escalation"] = consecutive_escalation
        details["reveal_density"] = reveal_density
        details["has_aftermath_recent"] = has_aftermath

        # ----- 弧线熵（arc）-----
        # 获取弧线信息
        if active_arcs is None and compressed_state is not None:
            active_arcs = getattr(compressed_state, 'character_arcs', {})
        if active_arcs is None:
            active_arcs = {}

        if not active_arcs:
            arc = 0.1
            unresolved = 0
            arc_escalation_penalty = 0.0
        else:
            unresolved = sum(1 for status in active_arcs.values() if status != "resolved")
            arc_unresolved_norm = min(unresolved / 10.0, 1.0)
            # 连续 escalation 对弧线熵的贡献
            arc_escalation_penalty = min(consecutive_escalation / 5.0, 0.5)
            arc = min(0.7 * arc_unresolved_norm + 0.3 * arc_escalation_penalty, 1.0)

        details["unresolved_arcs"] = unresolved
        details["arc_escalation_penalty"] = arc_escalation_penalty

        # ----- 文明熵（civilization）-----
        # 新设定引入率
        new_lore_rate = cls._compute_new_lore_rate(recent_events)
        # 境界膨胀惩罚（如果 world_state 有境界信息）
        realm_inflation = cls._compute_realm_inflation(world_state)
        # 势力/角色数量惩罚（粗略）
        character_count = len(world_state.characters) if hasattr(world_state, 'characters') else 0
        char_count_penalty = min(character_count / 50.0, 0.5)

        # 如果 recent_events 为空导致 new_lore_rate=0，则给一个基础值 0.05
        if new_lore_rate == 0.0 and not recent_events:
            new_lore_rate = 0.05

        civilization = min(0.5 * new_lore_rate + 0.3 * realm_inflation + 0.2 * char_count_penalty, 1.0)
        # 保证文明熵至少有一个极小的正值（避免全是0）
        if civilization == 0.0:
            civilization = 0.05

        details["new_lore_rate"] = new_lore_rate
        details["realm_inflation"] = realm_inflation
        details["character_count"] = character_count

        return EntropyReport(
            local=local,
            arc=arc,
            civilization=civilization,
            details=details,
        )

    # ==================== 旧版兼容方法 ====================

    @classmethod
    def calculate(
        cls,
        world_state: Any,
        arc_memory: Dict[str, Any],
        recent_events: List[Dict],
        compressed_state: Optional[Dict[str, Any]] = None,
    ) -> NarrativeEntropyScore:
        """
        旧版单值熵计算（保持向后兼容）
        """
        # 1. 未解决弧线数量
        unresolved_arcs = sum(1 for arc in arc_memory.values() if arc.get("status") != "resolved")

        # 2. 角色功能重叠度
        character_overlap = cls._compute_character_overlap(world_state, recent_events)

        # 3. 连续 escalation 场景数
        consecutive_escalation = 0
        for evt in reversed(recent_events):
            if evt.get("scene_role") == "ESCALATION":
                consecutive_escalation += 1
            else:
                break

        # 4. 新设定引入率
        new_lore_rate = cls._compute_new_lore_rate(recent_events)

        # 5. 未解决的关系数量
        unresolved_relationships = 0
        if hasattr(world_state, 'relationships') and isinstance(world_state.relationships, dict):
            unresolved_relationships = sum(1 for v in world_state.relationships.values() if isinstance(v, (int, float)) and v < -20)

        # 归一化
        max_unresolved = 10
        max_consecutive = 5
        max_unresolved_rel = 10

        normalized = {
            "unresolved_arcs": min(unresolved_arcs / max_unresolved, 1.0),
            "character_overlap": min(character_overlap, 1.0),
            "consecutive_escalation": min(consecutive_escalation / max_consecutive, 1.0),
            "new_lore_rate": min(new_lore_rate, 1.0),
            "unresolved_relationships": min(unresolved_relationships / max_unresolved_rel, 1.0),
        }

        total = sum(normalized[key] * cls.WEIGHTS[key] for key in cls.WEIGHTS.keys())

        return NarrativeEntropyScore(
            value=total,
            unresolved_arcs=unresolved_arcs,
            character_overlap=character_overlap,
            consecutive_escalation=consecutive_escalation,
            new_lore_rate=new_lore_rate,
            unresolved_relationships=unresolved_relationships,
        )

    # ==================== 辅助方法 ====================

    @classmethod
    def _compute_character_overlap(cls, world_state: Any, recent_events: List[Dict]) -> float:
        """计算角色功能重叠度（启发式）"""
        chars_in_events = set()
        for evt in recent_events:
            for char in evt.get("characters", []):
                chars_in_events.add(char)
        if len(chars_in_events) < 2:
            return 0.0

        tags_map = {}
        if hasattr(world_state, 'get_character_tags'):
            for char in chars_in_events:
                tags_map[char] = set(world_state.get_character_tags(char) or [])
        else:
            return 0.0

        total_jaccard = 0.0
        pairs = 0
        char_list = list(chars_in_events)
        for i in range(len(char_list)):
            for j in range(i + 1, len(char_list)):
                set_i = tags_map.get(char_list[i], set())
                set_j = tags_map.get(char_list[j], set())
                if not set_i and not set_j:
                    continue
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                if union > 0:
                    total_jaccard += intersection / union
                    pairs += 1
        return total_jaccard / pairs if pairs > 0 else 0.0

    @classmethod
    def _compute_new_lore_rate(cls, recent_events: List[Dict]) -> float:
        """计算新设定引入率"""
        if not recent_events:
            return 0.0
        new_lore_events = 0
        for evt in recent_events:
            if evt.get("event_type") in ("new_lore", "introduce_character", "discover_location", "acquire_item"):
                new_lore_events += 1
            elif evt.get("new_lore", False):
                new_lore_events += 1
        rate = new_lore_events / len(recent_events)
        return min(rate, 1.0)

    @classmethod
    def _compute_realm_inflation(cls, world_state: Any, protagonist_name: Optional[str] = None) -> float:
        """
        计算境界膨胀指数（粗略）
        
        Args:
            world_state: 世界状态对象
            protagonist_name: 可选，主角名称。若不提供则尝试自动推断。
        """
        if not hasattr(world_state, 'characters'):
            return 0.0
        
        # 获取主角
        protagonist = None
        if protagonist_name and protagonist_name in world_state.characters:
            protagonist = world_state.characters[protagonist_name]
        else:
            # 自动推断：查找名字含“林逸”或 tag 为 protagonist 的角色
            for name, char in world_state.characters.items():
                if "林逸" in name:
                    protagonist = char
                    break
                if hasattr(char, 'tags') and 'protagonist' in char.tags:
                    protagonist = char
                    break
            # 如果还没找到，取第一个角色作为主角
            if protagonist is None and world_state.characters:
                protagonist = next(iter(world_state.characters.values()))
        
        if protagonist is None:
            return 0.0

        # 从配置加载境界顺序
        cfg = get_xianxia_config()
        realm_order = cfg.rank.get("levels", ["凡人", "炼气", "筑基", "金丹", "元婴", "化神", "炼虚", "合体", "大乘", "渡劫"])

        current_realm = getattr(protagonist, 'realm', None)
        if current_realm is None:
            return 0.0
        realm_str = current_realm.value if hasattr(current_realm, 'value') else str(current_realm)
        try:
            idx = realm_order.index(realm_str)
        except ValueError:
            return 0.0
        # 境界越高，膨胀惩罚越大（假设超过元婴即开始通货膨胀）
        return min((idx - 3) / 6.0, 1.0) if idx > 3 else 0.0