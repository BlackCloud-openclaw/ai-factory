"""
感知传播服务 - 自动生成认知更新事件

当可观察事件发生时，为场景中所有在场角色生成 PerceptionUpdateEvent，
更新他们对事件相关方的认知关系。
"""

import logging
from typing import Dict, Any, List, Optional
from src.db import get_db_pool
from src.writing.event_store import NarrativeEventStore
from src.writing.world_state import WorldState
from src.writing.events import PerceptionUpdateEvent, NarrativeEvent
from src.config import config
from src.config_loader import get_xianxia_config

logger = logging.getLogger(__name__)

def _get_smoothing_factor() -> float:
    try:
        cfg = get_xianxia_config()
        return cfg.perception.get("smoothing_factor", 0.7)
    except:
        return 0.7

def _get_event_effect(event_type: str, effect_name: str) -> int:
    """从配置中获取事件感知效果值"""
    try:
        cfg = get_xianxia_config()
        effects = cfg.perception.get("event_effects", {})
        return effects.get(event_type, {}).get(effect_name, 0)
    except Exception:
        return 0


class PerceptionPropagationService:
    @staticmethod
    async def propagate(
        novel_id: str,
        volume: int,
        chapter: int,
        scene_idx: int,
        events: List[NarrativeEvent],
        world_state_before: WorldState,
    ) -> List[NarrativeEvent]:
        if not events:
            return []
        if not getattr(config, 'enable_perception_propagation', True):
            logger.debug("Perception propagation disabled")
            return []

        scene_location = world_state_before.map.current
        if not scene_location:
            logger.debug("No current location, cannot determine observers")
            return []

        present_characters = [
            name for name, char in world_state_before.characters.items()
            if char.location == scene_location
        ]
        if not present_characters:
            return []

        perception_events = []
        for event in events:
            updates = PerceptionPropagationService._process_event(event, present_characters, world_state_before)
            for update in updates:
                perception_event = PerceptionUpdateEvent(
                    observer=update["observer"],
                    target=update["target"],
                    new_value=update["new_value"],
                    confidence_delta=update.get("confidence_delta", 0.0),
                    reason=update.get("reason", ""),
                )
                perception_events.append(perception_event)
        if perception_events:
            logger.info(f"Generated {len(perception_events)} perception events")
        return perception_events

    @staticmethod
    def _process_event(event: NarrativeEvent, observers: List[str], world_state: WorldState) -> List[Dict]:
        event_type = getattr(event, 'type', None)
        if event_type is None:
            return []
        event_type_str = event_type.value if hasattr(event_type, 'value') else str(event_type)
        if event_type_str == "relationship_change":
            return PerceptionPropagationService._handle_relationship_change(event, observers, world_state)
        elif event_type_str == "combat_result":
            return PerceptionPropagationService._handle_combat_result(event, observers, world_state)
        elif event_type_str == "dialogue":
            return PerceptionPropagationService._handle_dialogue(event, observers, world_state)
        elif event_type_str == "realm_upgrade":
            return PerceptionPropagationService._handle_realm_upgrade(event, observers, world_state)
        elif event_type_str == "item_acquire":
            return PerceptionPropagationService._handle_item_acquire(event, observers, world_state)
        elif event_type_str == "location_enter":
            return PerceptionPropagationService._handle_location_enter(event, observers, world_state)
        return []

    @staticmethod
    def _handle_relationship_change(event, observers, world_state):
        results = []
        from_char = getattr(event, 'from_char', None)
        to_char = getattr(event, 'to_char', None)
        if not from_char or not to_char:
            return results
        
        # 获取平滑因子（只获取一次）
        smoothing = _get_smoothing_factor()
        
        for observer in observers:
            if observer in (from_char, to_char):
                continue
            obs_state = world_state.characters.get(observer)
            if not obs_state:
                continue
            # 对 from_char 的认知
            current_from = obs_state.perceived_relationships.get(from_char, {"value": 0, "confidence": 0.0})
            obj_from = world_state.relationships.get(f"{from_char}|{observer}", 0)
            new_val_from = int(current_from["value"] * smoothing + obj_from * (1 - smoothing))
            if new_val_from != current_from["value"]:
                results.append({
                    "observer": observer,
                    "target": from_char,
                    "new_value": new_val_from,
                    "confidence_delta": 0.2,
                    "reason": f"observed relationship change: {from_char} ↔ {to_char}",
                })
            # 对 to_char 的认知
            current_to = obs_state.perceived_relationships.get(to_char, {"value": 0, "confidence": 0.0})
            obj_to = world_state.relationships.get(f"{to_char}|{observer}", 0)
            new_val_to = int(current_to["value"] * smoothing + obj_to * (1 - smoothing))
            if new_val_to != current_to["value"]:
                results.append({
                    "observer": observer,
                    "target": to_char,
                    "new_value": new_val_to,
                    "confidence_delta": 0.2,
                    "reason": f"observed relationship change: {from_char} ↔ {to_char}",
                })
        return results

    @staticmethod
    def _handle_combat_result(event, observers, world_state):
        results = []
        winner = getattr(event, 'winner', None)
        loser = getattr(event, 'loser', None)
        if not winner or not loser:
            return results
        winner_effect = _get_event_effect("combat_result", "winner_effect") or 10
        loser_effect = _get_event_effect("combat_result", "loser_effect") or -10
        conf_delta = _get_event_effect("combat_result", "confidence_delta") or 0.3
        for observer in observers:
            if observer in (winner, loser):
                continue
            cur_win = world_state.characters.get(observer, {}).perceived_relationships.get(winner, {"value": 0, "confidence": 0.0})
            new_win = min(100, cur_win["value"] + winner_effect)
            if new_win != cur_win["value"]:
                results.append({
                    "observer": observer,
                    "target": winner,
                    "new_value": new_win,
                    "confidence_delta": conf_delta,
                    "reason": f"observed {winner} defeat {loser} in combat",
                })
            cur_lose = world_state.characters.get(observer, {}).perceived_relationships.get(loser, {"value": 0, "confidence": 0.0})
            new_lose = max(-100, cur_lose["value"] + loser_effect)
            if new_lose != cur_lose["value"]:
                results.append({
                    "observer": observer,
                    "target": loser,
                    "new_value": new_lose,
                    "confidence_delta": conf_delta,
                    "reason": f"observed {winner} defeat {loser} in combat",
                })
        return results

    @staticmethod
    def _handle_dialogue(event, observers, world_state):
        results = []
        speaker = getattr(event, 'speaker', None)
        listener = getattr(event, 'listener', None)
        key_revelation = getattr(event, 'key_revelation', None)
        if not speaker or not listener or not key_revelation:
            return results
        delta = -5 if any(kw in key_revelation for kw in ["秘密", "阴谋", "背叛"]) else 5
        for observer in observers:
            if observer in (speaker, listener):
                continue
            cur = world_state.characters.get(observer, {}).perceived_relationships.get(speaker, {"value": 0, "confidence": 0.0})
            new_val = min(100, max(-100, cur["value"] + delta))
            if new_val != cur["value"]:
                results.append({
                    "observer": observer,
                    "target": speaker,
                    "new_value": new_val,
                    "confidence_delta": 0.1,
                    "reason": f"overheard dialogue: {key_revelation[:30]}",
                })
        return results

    @staticmethod
    def _handle_realm_upgrade(event, observers, world_state):
        results = []
        actor = getattr(event, 'actor', None)
        if not actor:
            return results
        effect = _get_event_effect("realm_upgrade", "observer_effect") or 5
        conf_delta = _get_event_effect("realm_upgrade", "confidence_delta") or 0.1
        for observer in observers:
            if observer == actor:
                continue
            cur = world_state.characters.get(observer, {}).perceived_relationships.get(actor, {"value": 0, "confidence": 0.0})
            new_val = min(100, cur["value"] + effect)
            if new_val != cur["value"]:
                results.append({
                    "observer": observer,
                    "target": actor,
                    "new_value": new_val,
                    "confidence_delta": conf_delta,
                    "reason": f"observed {actor}'s realm upgrade",
                })
        return results

    @staticmethod
    def _handle_item_acquire(event, observers, world_state):
        results = []
        actor = getattr(event, 'actor', None)
        item = getattr(event, 'item', None)
        if not actor or not item:
            return results
        effect = _get_event_effect("item_acquire", "observer_effect") or 2
        conf_delta = _get_event_effect("item_acquire", "confidence_delta") or 0.05
        for observer in observers:
            if observer == actor:
                continue
            cur = world_state.characters.get(observer, {}).perceived_relationships.get(actor, {"value": 0, "confidence": 0.0})
            new_val = min(100, cur["value"] + effect)
            if new_val != cur["value"]:
                results.append({
                    "observer": observer,
                    "target": actor,
                    "new_value": new_val,
                    "confidence_delta": conf_delta,
                    "reason": f"observed {actor} acquired {item}",
                })
        return results

    @staticmethod
    def _handle_location_enter(event, observers, world_state):
        # 位置进入暂不产生感知变化
        return []