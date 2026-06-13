# src/writing/character_intent_memory.py
import json
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime


@dataclass
class CharacterIntentRecord:
    """角色的意图记录（长期存储）"""
    actor: str
    conscious_goal: str          # 显性目标
    hidden_need: str             # 深层需求
    fear: str                    # 恐惧
    misconception: Optional[str] = None
    immediate_tactic: str = ""
    last_updated: float = field(default_factory=datetime.now().timestamp)
    evolution_stage: int = 1     # 意图演变阶段（用于追踪变化）
    
    # ========== 认知身份（新增）==========
    beliefs: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)
    self_image: str = ""
    moral_boundaries: List[str] = field(default_factory=list)
    identity_evolution_stage: int = 1  # 身份演变阶段
    # ===================================


class CharacterIntentMemory:
    """角色意图的长期记忆管理"""
    
    def __init__(self, novel_id: str, compressed_state: Optional[Dict] = None):
        self.novel_id = novel_id
        self.intents: Dict[str, CharacterIntentRecord] = {}
        if compressed_state and "character_intents" in compressed_state:
            self._load_from_dict(compressed_state["character_intents"])
    
    def get_intent(self, actor: str) -> Optional[CharacterIntentRecord]:
        return self.intents.get(actor)
    
    def set_intent(self, actor: str, intent: CharacterIntentRecord):
        intent.last_updated = datetime.now().timestamp()
        self.intents[actor] = intent
    
    def update_from_director(self, director_intent: Dict[str, Any]):
        """从 Director 输出的 character_intent 更新记忆"""
        actor = director_intent.get("actor")
        if not actor:
            return
        existing = self.get_intent(actor)
        if existing:
            # 检查是否有显著变化
            changed = False
            if existing.conscious_goal != director_intent.get("conscious_goal"):
                existing.conscious_goal = director_intent.get("conscious_goal")
                changed = True
            if existing.hidden_need != director_intent.get("hidden_need"):
                existing.hidden_need = director_intent.get("hidden_need")
                changed = True
            if existing.fear != director_intent.get("fear"):
                existing.fear = director_intent.get("fear")
                changed = True
            if existing.misconception != director_intent.get("misconception"):
                existing.misconception = director_intent.get("misconception")
                changed = True
            if existing.immediate_tactic != director_intent.get("immediate_tactic", ""):
                existing.immediate_tactic = director_intent.get("immediate_tactic", "")
                changed = True
            
            # 认知身份变化检测
            new_beliefs = director_intent.get("beliefs")
            if new_beliefs and new_beliefs != existing.beliefs:
                existing.beliefs = new_beliefs
                existing.identity_evolution_stage += 1
                changed = True
                logger.info(f"Beliefs changed for {actor}: {new_beliefs}")
            
            new_attachments = director_intent.get("attachments")
            if new_attachments and new_attachments != existing.attachments:
                existing.attachments = new_attachments
                existing.identity_evolution_stage += 1
                changed = True
            
            new_self_image = director_intent.get("self_image")
            if new_self_image and new_self_image != existing.self_image:
                existing.self_image = new_self_image
                existing.identity_evolution_stage += 1
                changed = True
            
            new_moral_boundaries = director_intent.get("moral_boundaries")
            if new_moral_boundaries and new_moral_boundaries != existing.moral_boundaries:
                existing.moral_boundaries = new_moral_boundaries
                existing.identity_evolution_stage += 1
                changed = True
            
            if changed:
                existing.evolution_stage += 1
                existing.last_updated = datetime.now().timestamp()
        else:
            new_intent = CharacterIntentRecord(
                actor=actor,
                conscious_goal=director_intent.get("conscious_goal", ""),
                hidden_need=director_intent.get("hidden_need", ""),
                fear=director_intent.get("fear", ""),
                misconception=director_intent.get("misconception"),
                immediate_tactic=director_intent.get("immediate_tactic", ""),
                # 认知身份
                beliefs=director_intent.get("beliefs", []),
                attachments=director_intent.get("attachments", []),
                self_image=director_intent.get("self_image", ""),
                moral_boundaries=director_intent.get("moral_boundaries", []),
            )
            self.intents[actor] = new_intent
    
    def to_dict(self) -> Dict:
        return {actor: asdict(record) for actor, record in self.intents.items()}
    
    def _load_from_dict(self, data: Dict):
        for actor, record_dict in data.items():
            # 兼容旧格式
            if "last_updated" not in record_dict:
                record_dict["last_updated"] = datetime.now().timestamp()
            if "evolution_stage" not in record_dict:
                record_dict["evolution_stage"] = 1
            # 认知身份字段默认值
            if "beliefs" not in record_dict:
                record_dict["beliefs"] = []
            if "attachments" not in record_dict:
                record_dict["attachments"] = []
            if "self_image" not in record_dict:
                record_dict["self_image"] = ""
            if "moral_boundaries" not in record_dict:
                record_dict["moral_boundaries"] = []
            if "identity_evolution_stage" not in record_dict:
                record_dict["identity_evolution_stage"] = 1
            self.intents[actor] = CharacterIntentRecord(**record_dict)
    
    def get_all_intents_prompt(self) -> str:
        """生成用于 prompt 的角色意图总结"""
        if not self.intents:
            return ""
        lines = ["【角色意图记忆】"]
        for record in self.intents.values():
            lines.append(f"- {record.actor}: 目标={record.conscious_goal}, 恐惧={record.fear}")
            if record.hidden_need:
                lines.append(f"  深层需求={record.hidden_need}")
            if record.misconception:
                lines.append(f"  错误认知={record.misconception}")
            if record.evolution_stage > 1:
                lines.append(f"  已演变{record.evolution_stage}次")
            # 认知身份信息
            if record.self_image:
                lines.append(f"  自我认知={record.self_image}")
            if record.beliefs:
                lines.append(f"  核心信念={', '.join(record.beliefs[:3])}")
            if record.attachments:
                lines.append(f"  重要依恋={', '.join(record.attachments[:3])}")
        return "\n".join(lines)
    
    def get_intent_for_director_prompt(self, actor: str = "林逸") -> str:
        """获取单个角色的意图（用于 Director 的上下文）"""
        record = self.intents.get(actor)
        if not record:
            return ""
        parts = [f"当前{actor}的意图：显性目标={record.conscious_goal}，恐惧={record.fear}，深层需求={record.hidden_need}"]
        if record.self_image:
            parts.append(f"自我认知={record.self_image}")
        if record.beliefs:
            parts.append(f"核心信念={', '.join(record.beliefs[:3])}")
        return "；".join(parts)


# 添加 logger 导入（用于上面新增的日志）
import logging
logger = logging.getLogger(__name__)