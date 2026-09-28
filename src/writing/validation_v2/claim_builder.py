import hashlib
from typing import Optional, Dict, Any
from src.writing.planning_contract import StateChange
from .models import VerificationClaim


class ClaimBuilder:
    """从 StateChange 生成 VerificationClaim"""

    @classmethod
    def from_state_change(
        cls,
        state_change: StateChange,
        contract_id: str = "",
    ) -> VerificationClaim:
        raw_type = cls._get_raw_type(state_change)
        change_type = cls._normalize_type(raw_type)
        proposition, actor, target = cls._build_proposition(state_change, change_type)
        claim_id = cls._generate_id(contract_id, state_change.id or "", change_type)

        return VerificationClaim(
            claim_id=claim_id,
            state_change_type=change_type,
            raw_state_change_type=raw_type,
            proposition=proposition,
            actor=actor,
            target=target,
            source_contract_id=contract_id,
            source_state_change_id=state_change.id or "",
        )

    @classmethod
    def _get_raw_type(cls, change: StateChange) -> str:
        """获取原始类型，用于诊断"""
        t = getattr(change, "type", None)
        if t is None:
            return "None"
        if hasattr(t, "value"):
            return t.value
        return str(t)

    @classmethod
    def _normalize_type(cls, raw_type: str) -> str:
        """标准化类型，不强制消除 unknown"""
        if raw_type and raw_type != "None":
            return raw_type
        return "unknown"

    @classmethod
    def _build_proposition(
        cls,
        change: StateChange,
        change_type: str,
    ) -> tuple[str, Optional[str], Optional[Dict[str, Any]]]:
        actor = getattr(change, "actor", None)

        if change_type == "realm_change":
            target_realm = getattr(change, "to_major_realm", None)
            target_stage = getattr(change, "to_minor_stage", None)
            actor_name = actor or "角色"
            if target_stage and 1 <= target_stage <= 9:
                stage_chars = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]
                stage_str = stage_chars[target_stage - 1]
                proposition = f"{actor_name}已经突破至{target_realm}{stage_str}层"
            else:
                proposition = f"{actor_name}已经突破至{target_realm}"
            return proposition, actor, {"realm": target_realm, "stage": target_stage}

        if change_type == "plot_flag":
            name = getattr(change, "name", "未知标记")
            proposition = f"事件「{name}」已经发生"
            return proposition, None, {"flag_name": name}

        if change_type == "knowledge_gain":
            name = getattr(change, "name", "未知知识")
            actor_name = actor or "角色"
            proposition = f"{actor_name}已经获得了知识「{name}」"
            return proposition, actor, {"knowledge_name": name}

        if change_type == "inventory_acquire":
            item = getattr(change, "item", "未知物品")
            actor_name = actor or "角色"
            proposition = f"{actor_name}已经获得了物品「{item}」"
            return proposition, actor, {"item": item}

        if change_type == "location_change":
            location = getattr(change, "location", "未知地点")
            actor_name = actor or "角色"
            proposition = f"{actor_name}已经抵达「{location}」"
            return proposition, actor, {"location": location}

        if change_type == "relationship_change":
            from_char = getattr(change, "from_char", "A")
            to_char = getattr(change, "to_char", "B")
            delta = getattr(change, "delta", 0)
            if delta > 0:
                direction = "改善"
            elif delta < 0:
                direction = "恶化"
            else:
                direction = "变化"
            proposition = f"{from_char}与{to_char}的关系{direction}"
            return proposition, None, {"from": from_char, "to": to_char, "delta": delta}

        # fallback
        if actor:
            proposition = f"{actor}完成了{change_type}类型的状态变化"
        else:
            proposition = f"状态变化: {change_type}"
        return proposition, actor, None

    @classmethod
    def _generate_id(cls, contract_id: str, change_id: str, change_type: str) -> str:
        raw = f"{contract_id}|{change_id}|{change_type}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]