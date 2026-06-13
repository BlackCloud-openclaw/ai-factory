"""
叙事吸引子系统 - 定义叙事引力场

吸引子影响剧情走向，所有场景计划必须在其引力场内演化。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AttractorType(str, Enum):
    """吸引子类型"""
    PROTAGONIST = "protagonist"      # 主角吸引子
    THEMATIC = "thematic"            # 主题吸引子
    CONFLICT = "conflict"            # 冲突吸引子
    RELATIONSHIP = "relationship"    # 关系吸引子
    LOCATION = "location"            # 地点吸引子


@dataclass
class Attractor:
    """叙事吸引子"""
    id: str
    name: str
    type: AttractorType
    weight: float = 1.0              # 引力权重
    position: Optional[str] = None   # 位置（角色名/地点名/主题关键词）
    decay_distance: int = 5          # 衰减距离（章节）
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "weight": self.weight,
            "position": self.position,
            "decay_distance": self.decay_distance,
            "is_active": self.is_active,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Attractor":
        return cls(
            id=data["id"],
            name=data["name"],
            type=AttractorType(data["type"]),
            weight=data.get("weight", 1.0),
            position=data.get("position"),
            decay_distance=data.get("decay_distance", 5),
            is_active=data.get("is_active", True),
            metadata=data.get("metadata", {}),
        )


class NarrativeAttractorField:
    """叙事引力场 - 管理所有吸引子并计算引力"""
    
    def __init__(self):
        self.attractors: Dict[str, Attractor] = {}
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """初始化默认吸引子"""
        self.attractors["protagonist"] = Attractor(
            id="protagonist",
            name="主角引力场",
            type=AttractorType.PROTAGONIST,
            weight=2.0,
            position="林逸",
            decay_distance=999,  # 主角引力永不衰减
        )
        
        self.attractors["revenge"] = Attractor(
            id="revenge",
            name="复仇主题",
            type=AttractorType.THEMATIC,
            weight=1.5,
            position="复仇",
            decay_distance=10,
        )
        
        self.attractors["ascension"] = Attractor(
            id="ascension",
            name="飞升主题",
            type=AttractorType.THEMATIC,
            weight=1.5,
            position="飞升",
            decay_distance=10,
        )
    
    def register_attractor(self, attractor: Attractor):
        """注册新的吸引子"""
        self.attractors[attractor.id] = attractor
        logger.info(f"Registered attractor: {attractor.name} (weight={attractor.weight})")
    
    def remove_attractor(self, attractor_id: str):
        """移除吸引子"""
        if attractor_id in self.attractors:
            del self.attractors[attractor_id]
            logger.info(f"Removed attractor: {attractor_id}")
    
    def get_attractor(self, attractor_id: str) -> Optional[Attractor]:
        """获取吸引子"""
        return self.attractors.get(attractor_id)
    
    def calculate_gravity(
        self,
        scene_plan: Dict[str, Any],
        world_state: Any,
        current_chapter: int,
    ) -> float:
        """
        计算场景计划与所有吸引子的总引力值
        
        Returns:
            引力值（0-∞，越高表示越符合吸引子方向）
        """
        total_gravity = 0.0
        
        # 提取场景中的关键元素
        scene_characters = scene_plan.get("characters", [])
        scene_goal = scene_plan.get("goal", "")
        scene_conflict = scene_plan.get("conflict", "")
        scene_outcome = scene_plan.get("outcome", "")
        
        # 组合场景文本用于关键词匹配
        scene_text = f"{scene_goal} {scene_conflict} {scene_outcome}".lower()
        
        for attractor in self.attractors.values():
            if not attractor.is_active:
                continue
            
            # 计算该吸引子的贡献值
            contribution = self._calculate_attractor_contribution(
                attractor,
                scene_characters,
                scene_text,
                current_chapter,
            )
            total_gravity += contribution
        
        return total_gravity
    
    def _calculate_attractor_contribution(
        self,
        attractor: Attractor,
        scene_characters: List[str],
        scene_text: str,
        current_chapter: int,
    ) -> float:
        """计算单个吸引子的贡献"""
        
        if attractor.type == AttractorType.PROTAGONIST:
            # 主角吸引子：场景必须包含主角
            if attractor.position and attractor.position in scene_characters:
                return attractor.weight
            return 0.0
        
        elif attractor.type == AttractorType.THEMATIC:
            # 主题吸引子：关键词匹配
            if attractor.position:
                position_lower = attractor.position.lower()
                if position_lower in scene_text:
                    return attractor.weight
                # 部分匹配（主题词的部分出现在场景中）
                position_words = position_lower.split()
                if position_words and any(word in scene_text for word in position_words):
                    return attractor.weight * 0.5
            return 0.0
        
        elif attractor.type == AttractorType.CONFLICT:
            # 冲突吸引子：检查冲突类型匹配
            conflict_keywords = ["战斗", "争斗", "厮杀", "对决", "战争", "冲突", "对抗"]
            if any(kw in scene_text for kw in conflict_keywords):
                return attractor.weight
            return 0.0
        
        elif attractor.type == AttractorType.RELATIONSHIP:
            # 关系吸引子：检查相关角色是否出现
            if attractor.position and attractor.position in scene_characters:
                return attractor.weight
            return 0.0
        
        elif attractor.type == AttractorType.LOCATION:
            # 地点吸引子：检查地点是否出现
            if attractor.position and attractor.position in scene_text:
                return attractor.weight
            return 0.0
        
        return 0.0
    
    def get_gravity_score(self, scene_plan: Dict[str, Any], world_state: Any) -> Dict[str, float]:
        """获取每个吸引子的引力分数详情"""
        scores = {}
        scene_characters = scene_plan.get("characters", [])
        scene_text = f"{scene_plan.get('goal', '')} {scene_plan.get('conflict', '')}".lower()
        
        for attractor in self.attractors.values():
            if not attractor.is_active:
                continue
            
            if attractor.type == AttractorType.PROTAGONIST:
                score = attractor.weight if attractor.position in scene_characters else 0
            elif attractor.type == AttractorType.THEMATIC and attractor.position:
                score = attractor.weight if attractor.position.lower() in scene_text else 0
            elif attractor.type == AttractorType.CONFLICT:
                conflict_keywords = ["战斗", "争斗", "厮杀", "对决", "战争"]
                score = attractor.weight if any(kw in scene_text for kw in conflict_keywords) else 0
            else:
                score = 0
            scores[attractor.id] = score
        
        return scores
    
    def get_attractor_prompt(self, min_gravity: float = 0.5) -> str:
        """生成用于 Planner 的吸引子提示"""
        active_attractors = [a for a in self.attractors.values() if a.is_active]
        if not active_attractors:
            return ""
        
        lines = ["【🌟 叙事引力场约束】"]
        lines.append("以下主题和方向具有高引力值，场景计划应尽量靠近：")
        
        for attractor in active_attractors:
            if attractor.type == AttractorType.PROTAGONIST:
                lines.append(f"- 主角引力场（必须包含 {attractor.position}）权重 {attractor.weight}")
            elif attractor.type == AttractorType.THEMATIC:
                lines.append(f"- 主题引力场：围绕「{attractor.position}」展开 (权重 {attractor.weight})")
            elif attractor.type == AttractorType.CONFLICT:
                lines.append(f"- 冲突引力场：优先安排战斗/对抗场景 (权重 {attractor.weight})")
            elif attractor.type == AttractorType.RELATIONSHIP:
                lines.append(f"- 关系引力场：关注「{attractor.position}」 (权重 {attractor.weight})")
            elif attractor.type == AttractorType.LOCATION:
                lines.append(f"- 地点引力场：围绕「{attractor.position}」展开 (权重 {attractor.weight})")
        
        lines.append(f"\n⚠️ 场景计划的平均引力值低于 {min_gravity} 时，会被退回重规划。")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化"""
        return {
            "attractors": {
                aid: a.to_dict()
                for aid, a in self.attractors.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NarrativeAttractorField":
        """反序列化"""
        field = cls()
        field.attractors.clear()
        for aid, a_data in data.get("attractors", {}).items():
            field.attractors[aid] = Attractor.from_dict(a_data)
        return field