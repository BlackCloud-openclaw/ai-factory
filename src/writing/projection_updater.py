"""
Phase 13.2: ProjectionUpdater

确定性 Reducer：根据 NarrativeIntent + Runtime Events 更新 NarrativeProjection。
这是纯函数，不包含 LLM 调用或外部状态。

核心原则：
- 输入不可变（previous projection 不被修改）
- 输出为新 projection（version + 1）
- 所有更新操作返回 model_copy
"""

from typing import Optional, List
from src.writing.narrative_projection import NarrativeProjection
from src.writing.narrative_intent import NarrativeIntent, SceneRole
from src.writing.events import NarrativeEvent
import logging

logger = logging.getLogger(__name__)

# importance 映射（用于 DiscoveryEvent）
IMPORTANCE_MAP = {"low": 1, "normal": 2, "high": 3, "critical": 4}


class ProjectionUpdater:
    """
    确定性投影更新器。

    输入：
        - previous_projection: NarrativeProjection（可为 None）
        - intent: NarrativeIntent（当前场景意图）
        - events: List[NarrativeEvent]（当前场景产生的事件）

    输出：
        - new_projection: NarrativeProjection（version + 1）
    """

    def update(
        self,
        previous: Optional[NarrativeProjection],
        intent: NarrativeIntent,
        events: List[NarrativeEvent]
    ) -> NarrativeProjection:
        """执行确定性更新 - 纯函数，不修改输入对象"""
        # 1. 初始状态
        if previous is None:
            projection = self._create_default(intent)
        else:
            projection = previous.increment_version()

        # 2. 所有更新方法都返回新对象
        projection = self._update_active_conflict(projection, intent, events)
        projection = self._update_threads(projection, intent, events)
        projection = self._update_objectives(projection, intent, events)
        projection = self._update_emotional_state(projection, intent, events)
        projection = self._update_next_pressure(projection, intent, events)

        # 3. 更新元数据（使用 model_copy 而非直接赋值）
        projection = projection.model_copy(
            update={
                "last_intent_id": intent.intent_id,
                "last_scene_role": intent.scene_role,
            }
        )

        logger.debug(
            f"[ProjectionUpdater] Updated projection to version {projection.version}: "
            f"conflict={projection.active_conflict}, threads={len(projection.unresolved_threads)}"
        )

        return projection

    def _create_default(self, intent: NarrativeIntent) -> NarrativeProjection:
        """
        创建初始 Projection（当无 previous 时）。

        优先级：
        1. 从 consequences 中提取 conflict_established 事件
        2. 如果场景是冲突类角色，fallback 到 objective
        3. 否则为 None
        """
        active_conflict = None

        # 1. 从 consequences 中提取明确的冲突
        for c in intent.consequences:
            if c.event_type == "conflict_established":
                active_conflict = c.target
                break

        # 2. fallback：冲突类角色使用 objective
        if active_conflict is None:
            conflict_roles = {
                SceneRole.CONFLICT_ESCALATION,
                SceneRole.CONFRONTATION,
                SceneRole.CLIMAX,
            }
            if intent.scene_role in conflict_roles:
                active_conflict = intent.objective

        return NarrativeProjection(
            projection_id=NarrativeProjection.generate_projection_id(
                chapter_id="initial",
                last_intent_id=intent.intent_id
            ),
            chapter_id="initial",
            active_conflict=active_conflict,
            unresolved_threads=[],
            active_objectives=[intent.objective],
            emotional_state="neutral",
            next_pressure=None,
            last_intent_id=intent.intent_id,
            last_scene_role=intent.scene_role,
            version=1,
        )

    def _update_active_conflict(
        self,
        projection: NarrativeProjection,
        intent: NarrativeIntent,
        events: List[NarrativeEvent]
    ) -> NarrativeProjection:
        """
        根据场景角色和事件更新 active_conflict。
        如果场景角色是冲突升级类，且事件中有冲突相关事件，则更新。
        """
        conflict_roles = {
            SceneRole.CONFLICT_ESCALATION,
            SceneRole.CONFRONTATION,
            SceneRole.CLIMAX_PREPARATION,
        }
        if intent.scene_role in conflict_roles:
            # 检查事件中是否有关系变化或战斗结果
            has_conflict = any(
                e.type.value in ("relationship_change", "combat_result")
                for e in events
            )
            if has_conflict:
                new_conflict = intent.objective
                if projection.active_conflict:
                    if new_conflict not in projection.active_conflict:
                        return projection.model_copy(
                            update={"active_conflict": f"{projection.active_conflict} → {new_conflict}"}
                        )
                else:
                    return projection.model_copy(
                        update={"active_conflict": new_conflict}
                    )
        return projection

    def _update_threads(
        self,
        projection: NarrativeProjection,
        intent: NarrativeIntent,
        events: List[NarrativeEvent]
    ) -> NarrativeProjection:
        """
        更新 unresolved_threads：
        - 从 consequences 中添加新线索
        - 从 events 中移除已解决的线索
        """
        # 添加新线索：从 consequences 中提取 target
        new_threads = []
        for c in intent.consequences:
            if c.target.startswith("knowledge.") or c.target.startswith("quest."):
                if c.operation == "set" and c.value is True:
                    thread_desc = c.target.replace("knowledge.", "").replace("quest.", "")
                    if thread_desc not in projection.unresolved_threads:
                        new_threads.append(thread_desc)

        # 移除已解决的线索：从 events 中查找对应的 discovery 或 plot_flag_set
        resolved = set()
        importance_map = IMPORTANCE_MAP

        for e in events:
            if e.type.value == "discovery":
                discovery = getattr(e, 'discovery', '')
                if discovery:
                    importance_val = getattr(e, 'importance', 'normal')
                    importance_num = importance_map.get(importance_val, 0)
                    for thread in projection.unresolved_threads:
                        if thread in discovery and importance_num >= 3:
                            resolved.add(thread)
            elif e.type.value == "plot_flag_set":
                flag = getattr(e, 'flag', '')
                if flag.startswith("solved_"):
                    thread = flag.replace("solved_", "")
                    if thread in projection.unresolved_threads:
                        resolved.add(thread)

        current = set(projection.unresolved_threads)
        current.update(new_threads)
        current.difference_update(resolved)

        return projection.model_copy(
            update={"unresolved_threads": list(current)}
        )

    def _update_objectives(
        self,
        projection: NarrativeProjection,
        intent: NarrativeIntent,
        events: List[NarrativeEvent]
    ) -> NarrativeProjection:
        """
        根据 intent.objective 和 consequences 更新 active_objectives。
        """
        new_objectives = [intent.objective]
        for c in intent.consequences:
            if c.target.startswith("quest."):
                if c.operation == "set" and c.value is True:
                    quest = c.target.replace("quest.", "")
                    if quest not in new_objectives:
                        new_objectives.append(quest)

        importance_map = IMPORTANCE_MAP
        for e in events:
            if e.type.value == "discovery":
                importance_val = getattr(e, 'importance', 'normal')
                importance_num = importance_map.get(importance_val, 0)
                if importance_num >= 5:  # critical
                    discovery = getattr(e, 'discovery', '')
                    if discovery and discovery not in new_objectives:
                        new_objectives.append(f"调查: {discovery}")

        return projection.model_copy(
            update={"active_objectives": new_objectives[:5]}
        )

    def _update_emotional_state(
        self,
        projection: NarrativeProjection,
        intent: NarrativeIntent,
        events: List[NarrativeEvent]
    ) -> NarrativeProjection:
        """
        根据场景角色和事件更新 emotional_state。
        当前为 deterministic derivation，非 authoritative state。
        （Phase 13.3 可能升级）
        """
        role_to_emotion = {
            SceneRole.SETUP: "好奇",
            SceneRole.TRANSITION: "平静",
            SceneRole.DISCOVERY: "震惊",
            SceneRole.CONFLICT_ESCALATION: "紧张",
            SceneRole.CONFRONTATION: "愤怒",
            SceneRole.CHARACTER_DECISION: "挣扎",
            SceneRole.CONSEQUENCE: "沉重",
            SceneRole.RECOVERY: "放松",
            SceneRole.CLIMAX_PREPARATION: "期待",
            SceneRole.CLIMAX: "激动",
            SceneRole.RESOLUTION: "释然",
        }
        base_emotion = role_to_emotion.get(intent.scene_role, "中性")

        for e in events:
            if e.type.value == "relationship_change":
                delta = getattr(e, 'delta', 0)
                if delta < 0:
                    base_emotion = f"{base_emotion}，信任下降"
                elif delta > 0:
                    base_emotion = f"{base_emotion}，信任上升"

        return projection.model_copy(
            update={"emotional_state": base_emotion}
        )

    def _update_next_pressure(
        self,
        projection: NarrativeProjection,
        intent: NarrativeIntent,
        events: List[NarrativeEvent]
    ) -> NarrativeProjection:
        """
        根据 unresolved_threads 和 active_conflict 生成 next_pressure。
        当前为 deterministic derivation，非 authoritative state。
        """
        if not projection.unresolved_threads and not projection.active_conflict:
            return projection

        if projection.unresolved_threads:
            next_pressure = f"必须解决: {projection.unresolved_threads[0]}"
        else:
            next_pressure = f"推进: {projection.active_conflict}"

        if len(projection.unresolved_threads) > 1:
            next_pressure = f"多重压力: {', '.join(projection.unresolved_threads[:2])}..."

        return projection.model_copy(
            update={"next_pressure": next_pressure}
        )