"""
感知传播服务 - 自动生成认知更新事件

当可观察事件发生时，为场景中所有在场角色生成 PerceptionUpdateEvent，
更新他们对事件相关方的认知关系。
"""

import logging
from typing import Dict, Any, List, Set, Optional
from src.db import get_db_pool
from src.writing.event_store import NarrativeEventStore
from src.writing.world_state import WorldState
from src.writing.events import PerceptionUpdateEvent, NarrativeEvent
from src.config import config

logger = logging.getLogger(__name__)


class PerceptionPropagationService:
    """
    感知传播服务：根据事件和在场角色，自动生成认知更新事件
    """

    @staticmethod
    async def propagate(
        novel_id: str,
        volume: int,
        chapter: int,
        scene_idx: int,
        events: List[NarrativeEvent],
        world_state_before: WorldState,
    ) -> List[NarrativeEvent]:
        """
        处理事件列表，生成新的感知更新事件

        Args:
            novel_id: 小说ID
            volume: 卷号
            chapter: 章号
            scene_idx: 场景索引
            events: 已发生的事件列表（原始事件）
            world_state_before: 事件发生前的世界状态（用于确定在场角色）

        Returns:
            新生成的 PerceptionUpdateEvent 列表
        """
        if not events:
            return []

        # 检查配置开关
        if not getattr(config, 'enable_perception_propagation', True):
            logger.debug("Perception propagation disabled by config")
            return []

        # 确定场景中所有在场角色（基于 location）
        scene_location = world_state_before.map.current
        if not scene_location:
            logger.debug("No current location, cannot determine observers")
            return []

        present_characters = [
            name for name, char in world_state_before.characters.items()
            if char.location == scene_location
        ]
        
        if not present_characters:
            logger.debug(f"No characters present at location {scene_location}")
            return []

        logger.info(f"Perception propagation: {len(present_characters)} observers at {scene_location}, processing {len(events)} events")

        perception_events = []

        for event in events:
            # 根据事件类型生成感知更新
            updates = PerceptionPropagationService._process_event(
                event, present_characters, world_state_before
            )
            for update in updates:
                perception_event = PerceptionUpdateEvent(
                    observer=update["observer"],
                    target=update["target"],
                    new_value=update["new_value"],
                    confidence_delta=update.get("confidence_delta", 0.0),
                    reason=update.get("reason", ""),
                )
                # 不需要手动设置 novel_id 等，append_event 时会作为参数传入
                perception_events.append(perception_event)                
                logger.debug(f"Generated perception: {update['observer']} -> {update['target']} = {update['new_value']} (Δconf={update.get('confidence_delta', 0)})")

        if perception_events:
            logger.info(f"Generated {len(perception_events)} perception events from {len(events)} original events")

        return perception_events

    @staticmethod
    def _process_event(
        event: NarrativeEvent,
        observers: List[str],
        world_state: WorldState,
    ) -> List[Dict[str, Any]]:
        """
        根据事件类型，为每个观察者生成认知更新指令
        返回格式: [{"observer": str, "target": str, "new_value": int, "confidence_delta": float, "reason": str}]
        """
        results = []
        event_type = getattr(event, 'type', None)
        if event_type is None:
            return results

        event_type_str = event_type.value if hasattr(event_type, 'value') else str(event_type)

        # 根据事件类型分发
        if event_type_str == "relationship_change":
            results = PerceptionPropagationService._handle_relationship_change(
                event, observers, world_state
            )
        elif event_type_str == "combat_result":
            results = PerceptionPropagationService._handle_combat_result(
                event, observers, world_state
            )
        elif event_type_str == "dialogue":
            results = PerceptionPropagationService._handle_dialogue(
                event, observers, world_state
            )
        elif event_type_str == "realm_upgrade":
            results = PerceptionPropagationService._handle_realm_upgrade(
                event, observers, world_state
            )
        elif event_type_str == "item_acquire":
            results = PerceptionPropagationService._handle_item_acquire(
                event, observers, world_state
            )
        elif event_type_str == "location_enter":
            results = PerceptionPropagationService._handle_location_enter(
                event, observers, world_state
            )

        return results

    # ========== 事件类型处理器 ==========

    @staticmethod
    def _handle_relationship_change(
        event: NarrativeEvent,
        observers: List[str],
        world_state: WorldState,
    ) -> List[Dict[str, Any]]:
        """处理关系变化事件"""
        results = []
        from_char = getattr(event, 'from_char', None)
        to_char = getattr(event, 'to_char', None)
        delta = getattr(event, 'delta', 0)
        if not from_char or not to_char:
            return results

        # 确定新值（如果事件中有 new_value 字段）
        new_value = getattr(event, 'new_value', None)
        
        for observer in observers:
            # 观察者是自己时，已经直接经历，无需额外更新
            if observer == from_char or observer == to_char:
                continue

            # 获取观察者的当前认知状态
            observer_state = world_state.characters.get(observer)
            if not observer_state:
                continue

            # 更新对 from_char 的认知
            current_from = observer_state.perceived_relationships.get(
                from_char, {"value": 0, "confidence": 0.0}
            )
            # 认知值向客观值移动（平滑因子 0.3）
            objective_from = world_state.relationships.get(f"{from_char}|{observer}", 0)
            new_value_from = int(current_from["value"] * 0.7 + objective_from * 0.3)
            confidence_delta_from = 0.2
            
            # 只有真正变化时才生成事件
            if new_value_from != current_from["value"]:
                results.append({
                    "observer": observer,
                    "target": from_char,
                    "new_value": new_value_from,
                    "confidence_delta": confidence_delta_from,
                    "reason": f"observed relationship change: {from_char} ↔ {to_char}",
                })

            # 更新对 to_char 的认知
            current_to = observer_state.perceived_relationships.get(
                to_char, {"value": 0, "confidence": 0.0}
            )
            objective_to = world_state.relationships.get(f"{to_char}|{observer}", 0)
            new_value_to = int(current_to["value"] * 0.7 + objective_to * 0.3)
            confidence_delta_to = 0.2
            
            if new_value_to != current_to["value"]:
                results.append({
                    "observer": observer,
                    "target": to_char,
                    "new_value": new_value_to,
                    "confidence_delta": confidence_delta_to,
                    "reason": f"observed relationship change: {from_char} ↔ {to_char}",
                })

        return results

    @staticmethod
    def _handle_combat_result(
        event: NarrativeEvent,
        observers: List[str],
        world_state: WorldState,
    ) -> List[Dict[str, Any]]:
        """处理战斗结果事件"""
        results = []
        winner = getattr(event, 'winner', None)
        loser = getattr(event, 'loser', None)
        if not winner or not loser:
            return results

        for observer in observers:
            if observer == winner or observer == loser:
                continue

            # 对胜者好感 +10（认知值增加）
            current_winner = world_state.characters.get(observer, {}).perceived_relationships.get(
                winner, {"value": 0, "confidence": 0.0}
            )
            new_winner_value = min(100, current_winner["value"] + 10)
            if new_winner_value != current_winner["value"]:
                results.append({
                    "observer": observer,
                    "target": winner,
                    "new_value": new_winner_value,
                    "confidence_delta": 0.3,
                    "reason": f"observed {winner} defeat {loser} in combat",
                })

            # 对败者好感 -10
            current_loser = world_state.characters.get(observer, {}).perceived_relationships.get(
                loser, {"value": 0, "confidence": 0.0}
            )
            new_loser_value = max(-100, current_loser["value"] - 10)
            if new_loser_value != current_loser["value"]:
                results.append({
                    "observer": observer,
                    "target": loser,
                    "new_value": new_loser_value,
                    "confidence_delta": 0.3,
                    "reason": f"observed {winner} defeat {loser} in combat",
                })

        return results

    @staticmethod
    def _handle_dialogue(
        event: NarrativeEvent,
        observers: List[str],
        world_state: WorldState,
    ) -> List[Dict[str, Any]]:
        """处理对话事件"""
        results = []
        speaker = getattr(event, 'speaker', None)
        listener = getattr(event, 'listener', None)
        key_revelation = getattr(event, 'key_revelation', None)

        if not speaker or not listener:
            return results

        for observer in observers:
            # 对话参与者不需要额外更新
            if observer == speaker or observer == listener:
                continue

            # 如果对话包含关键信息，观察者可能调整对 speaker 的认知
            if key_revelation:
                current_speaker = world_state.characters.get(observer, {}).perceived_relationships.get(
                    speaker, {"value": 0, "confidence": 0.0}
                )
                # 根据关键信息内容粗略调整（这里简单 +5 或 -5）
                if "秘密" in key_revelation or "阴谋" in key_revelation or "背叛" in key_revelation:
                    delta = -5
                else:
                    delta = 5
                new_speaker_value = min(100, max(-100, current_speaker["value"] + delta))
                if new_speaker_value != current_speaker["value"]:
                    results.append({
                        "observer": observer,
                        "target": speaker,
                        "new_value": new_speaker_value,
                        "confidence_delta": 0.1,
                        "reason": f"overheard dialogue: {key_revelation[:30]}",
                    })

        return results

    @staticmethod
    def _handle_realm_upgrade(
        event: NarrativeEvent,
        observers: List[str],
        world_state: WorldState,
    ) -> List[Dict[str, Any]]:
        """处理境界突破事件"""
        results = []
        actor = getattr(event, 'actor', None)
        if not actor:
            return results

        for observer in observers:
            if observer == actor:
                continue
            # 观察到别人突破，一般会提升敬畏度（正向）
            current = world_state.characters.get(observer, {}).perceived_relationships.get(
                actor, {"value": 0, "confidence": 0.0}
            )
            new_value = min(100, current["value"] + 5)
            if new_value != current["value"]:
                results.append({
                    "observer": observer,
                    "target": actor,
                    "new_value": new_value,
                    "confidence_delta": 0.1,
                    "reason": f"observed {actor}'s realm upgrade",
                })

        return results

    @staticmethod
    def _handle_item_acquire(
        event: NarrativeEvent,
        observers: List[str],
        world_state: WorldState,
    ) -> List[Dict[str, Any]]:
        """处理获得物品事件"""
        results = []
        actor = getattr(event, 'actor', None)
        item = getattr(event, 'item', None)
        if not actor or not item:
            return results

        for observer in observers:
            if observer == actor:
                continue
            # 观察到别人获得宝物，好感可能微增（羡慕或嫉妒，这里简单+2）
            current = world_state.characters.get(observer, {}).perceived_relationships.get(
                actor, {"value": 0, "confidence": 0.0}
            )
            new_value = min(100, current["value"] + 2)
            if new_value != current["value"]:
                results.append({
                    "observer": observer,
                    "target": actor,
                    "new_value": new_value,
                    "confidence_delta": 0.05,
                    "reason": f"observed {actor} acquired {item}",
                })

        return results

    @staticmethod
    def _handle_location_enter(
        event: NarrativeEvent,
        observers: List[str],
        world_state: WorldState,
    ) -> List[Dict[str, Any]]:
        """处理进入地点事件"""
        results = []
        actor = getattr(event, 'actor', None)
        location = getattr(event, 'location', None)
        if not actor or not location:
            return results

        for observer in observers:
            if observer == actor:
                continue
            # 观察到别人进入禁地/密地，好感可能微调（好奇或警惕）
            # 这里暂时不处理，因为影响较小
            pass

        return results