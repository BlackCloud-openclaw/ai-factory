"""
相变系统 - 检测和处理叙事中的不可逆质变

当系统变量超过阈值时，触发相变事件，影响后续行为。
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from .world_state import WorldState
from .memory_hierarchy import CompressedState

logger = logging.getLogger(__name__)


class PhaseTransitionType(str, Enum):
    """相变类型"""
    RELATIONSHIP_COLLAPSE = "relationship_collapse"  # 关系崩塌
    IDENTITY_SHIFT = "identity_shift"                # 身份转变
    REALM_ASCENSION = "realm_ascension"              # 境界飞升
    WORLD_CONFLICT = "world_conflict"                # 世界级冲突
    THEMATIC_CONVERGENCE = "thematic_convergence"    # 主题收敛


@dataclass
class PhaseTransition:
    """相变记录"""
    type: PhaseTransitionType
    triggered_at: float = field(default_factory=datetime.now().timestamp)
    details: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    resolved_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "triggered_at": self.triggered_at,
            "details": self.details,
            "is_active": self.is_active,
            "resolved_at": self.resolved_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseTransition":
        return cls(
            type=PhaseTransitionType(data["type"]),
            triggered_at=data["triggered_at"],
            details=data.get("details", {}),
            is_active=data.get("is_active", True),
            resolved_at=data.get("resolved_at"),
        )


class PhaseTransitionDetector:
    """相变检测器"""
    
    # 阈值配置
    THRESHOLDS = {
        "relationship_collapse_value": -80,
        "relationship_collapse_confidence": 0.7,
        "identity_shift_threshold": 3,  # 身份演变阶段变化次数
        "realm_ascension_cooldown_chapters": 2,
        "world_conflict_entropy": 0.85,
    }
    
    @classmethod
    def detect(
        cls,
        world_state: WorldState,
        compressed_state: Optional[CompressedState],
        previous_transitions: List[PhaseTransition],
    ) -> List[PhaseTransition]:
        """
        检测所有可能的相变
        
        Returns:
            新触发的相变列表
        """
        new_transitions = []
        
        # 1. 关系崩塌检测
        collapse = cls._detect_relationship_collapse(world_state, previous_transitions)
        if collapse:
            new_transitions.append(collapse)
        
        # 2. 身份转变检测
        identity = cls._detect_identity_shift(compressed_state, previous_transitions)
        if identity:
            new_transitions.append(identity)
        
        # 3. 境界飞升检测
        ascension = cls._detect_realm_ascension(world_state, previous_transitions)
        if ascension:
            new_transitions.append(ascension)
        
        # 4. 世界冲突检测
        conflict = cls._detect_world_conflict(compressed_state, previous_transitions)
        if conflict:
            new_transitions.append(conflict)
        
        if new_transitions:
            logger.info(f"Detected {len(new_transitions)} phase transitions: {[t.type.value for t in new_transitions]}")
        
        return new_transitions
    
    @classmethod
    def _detect_relationship_collapse(
        cls,
        world_state: WorldState,
        previous_transitions: List[PhaseTransition],
    ) -> Optional[PhaseTransition]:
        """检测关系崩塌"""
        # 检查是否已经触发过且未解决
        for pt in previous_transitions:
            if pt.type == PhaseTransitionType.RELATIONSHIP_COLLAPSE and pt.is_active:
                return None
        
        # 遍历所有关系
        for key, value in world_state.relationships.items():
            if value <= cls.THRESHOLDS["relationship_collapse_value"]:
                # 检查确信度（需要从角色认知中获取）
                parts = key.split("|")
                if len(parts) == 2:
                    a, b = parts
                    # 获取 A 对 B 的认知确信度
                    if a in world_state.characters:
                        perception = world_state.characters[a].perceived_relationships.get(b, {})
                        confidence = perception.get("confidence", 0.0)
                        if confidence >= cls.THRESHOLDS["relationship_collapse_confidence"]:
                            logger.info(f"Relationship collapse detected: {key} = {value}, confidence={confidence}")
                            return PhaseTransition(
                                type=PhaseTransitionType.RELATIONSHIP_COLLAPSE,
                                details={
                                    "relationship": key,
                                    "value": value,
                                    "confidence": confidence,
                                    "actors": [a, b],
                                }
                            )
        return None
    
    @classmethod
    def _detect_identity_shift(
        cls,
        compressed_state: Optional[CompressedState],
        previous_transitions: List[PhaseTransition],
    ) -> Optional[PhaseTransition]:
        """检测身份转变"""
        if not compressed_state:
            return None
        
        # 检查是否已经触发过且未解决
        for pt in previous_transitions:
            if pt.type == PhaseTransitionType.IDENTITY_SHIFT and pt.is_active:
                return None
        
        # 从 character_intents 中获取演变阶段
        character_intents = getattr(compressed_state, 'character_intents', {})
        for actor, intent in character_intents.items():
            evolution_stage = intent.get("identity_evolution_stage", 0)
            if evolution_stage >= cls.THRESHOLDS["identity_shift_threshold"]:
                logger.info(f"Identity shift detected for {actor}: evolution_stage={evolution_stage}")
                return PhaseTransition(
                    type=PhaseTransitionType.IDENTITY_SHIFT,
                    details={
                        "actor": actor,
                        "evolution_stage": evolution_stage,
                        "beliefs": intent.get("beliefs", []),
                        "self_image": intent.get("self_image", ""),
                    }
                )
        return None
    
    @classmethod
    def _detect_realm_ascension(
        cls,
        world_state: WorldState,
        previous_transitions: List[PhaseTransition],
    ) -> Optional[PhaseTransition]:
        """检测境界飞升"""
        # 检查是否有飞升冷却
        if world_state.global_flags.get("realm_ascension_cooldown", False):
            remaining = world_state.global_flags.get("realm_ascension_cooldown_remaining", 0)
            if remaining > 0:
                return None
        
        # 检测主角境界
        protagonist = "林逸"
        if protagonist in world_state.characters:
            char = world_state.characters[protagonist]
            # 跨大境界突破检测（金丹及以上视为重要突破）
            major_ascension_realms = ["金丹", "元婴", "化神", "炼虚", "合体", "大乘", "渡劫"]
            if char.realm.value in major_ascension_realms and char.realm_level == 1:
                logger.info(f"Realm ascension detected: {char.full_realm()}")
                return PhaseTransition(
                    type=PhaseTransitionType.REALM_ASCENSION,
                    details={
                        "actor": protagonist,
                        "new_realm": char.full_realm(),
                        "realm_value": char.realm.value,
                        "level": char.realm_level,
                    }
                )
        return None
    
    @classmethod
    def _detect_world_conflict(
        cls,
        compressed_state: Optional[CompressedState],
        previous_transitions: List[PhaseTransition],
    ) -> Optional[PhaseTransition]:
        """检测世界级冲突"""
        if not compressed_state:
            return None
        
        # 检查是否已经触发
        for pt in previous_transitions:
            if pt.type == PhaseTransitionType.WORLD_CONFLICT and pt.is_active:
                return None
        
        # 使用文明熵
        civ_entropy = getattr(compressed_state, 'civilization_entropy', 0.0)
        if civ_entropy >= cls.THRESHOLDS["world_conflict_entropy"]:
            logger.info(f"World conflict detected: civilization_entropy={civ_entropy}")
            return PhaseTransition(
                type=PhaseTransitionType.WORLD_CONFLICT,
                details={
                    "civilization_entropy": civ_entropy,
                    "reason": "Entropy threshold exceeded",
                }
            )
        return None


class PhaseTransitionHandler:
    """相变处理器 - 处理相变后的系统调整"""
    
    @staticmethod
    def apply_transition(
        transition: PhaseTransition,
        world_state: WorldState,
    ) -> WorldState:
        """应用相变到世界状态"""
        from copy import deepcopy
        
        new_state = deepcopy(world_state)
        
        if transition.type == PhaseTransitionType.RELATIONSHIP_COLLAPSE:
            details = transition.details
            rel_key = details.get("relationship")
            if rel_key and rel_key in new_state.relationships:
                # 标记关系为不可逆
                new_state.relationships[rel_key] = -100
                # 添加全局标记
                new_state.global_flags[f"collapsed_{rel_key.replace('|', '_')}"] = True
                logger.info(f"Applied relationship collapse: {rel_key}")
                
        elif transition.type == PhaseTransitionType.IDENTITY_SHIFT:
            actor = transition.details.get("actor")
            if actor and actor in new_state.characters:
                # 记录身份转变
                if "identity_transitions" not in new_state.global_flags:
                    new_state.global_flags["identity_transitions"] = []
                identity_transitions = new_state.global_flags["identity_transitions"]
                if isinstance(identity_transitions, list):
                    identity_transitions.append({
                        "actor": actor,
                        "timestamp": transition.triggered_at,
                        "details": transition.details,
                    })
                logger.info(f"Applied identity shift for {actor}")
                
        elif transition.type == PhaseTransitionType.REALM_ASCENSION:
            # 添加飞升冷却标记
            new_state.global_flags["realm_ascension_cooldown"] = True
            new_state.global_flags["realm_ascension_cooldown_remaining"] = \
                PhaseTransitionDetector.THRESHOLDS["realm_ascension_cooldown_chapters"]
            logger.info(f"Applied realm ascension cooldown")
                
        elif transition.type == PhaseTransitionType.WORLD_CONFLICT:
            new_state.global_flags["world_conflict_mode"] = True
            logger.info(f"Applied world conflict mode")
            
        return new_state
    
    @staticmethod
    def decrement_cooldowns(world_state: WorldState) -> WorldState:
        """减少冷却计数器（每章结束时调用）"""
        from copy import deepcopy
        
        new_state = deepcopy(world_state)
        
        # 减少飞升冷却
        if new_state.global_flags.get("realm_ascension_cooldown", False):
            remaining = new_state.global_flags.get("realm_ascension_cooldown_remaining", 0)
            if remaining <= 1:
                new_state.global_flags["realm_ascension_cooldown"] = False
                new_state.global_flags.pop("realm_ascension_cooldown_remaining", None)
                logger.info("Realm ascension cooldown ended")
            else:
                new_state.global_flags["realm_ascension_cooldown_remaining"] = remaining - 1
        
        return new_state