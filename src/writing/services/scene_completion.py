# src/writing/services/scene_completion.py
import asyncpg
from src.db import get_db_pool
from src.writing.event_store import NarrativeEventStore
from src.writing.snapshot import SnapshotManager
from src.writing.delta import StateDelta
from src.writing.world_state import WorldState
from src.writing.events import event_from_dict
from src.orchestrator.state_patch import StatePatch, WorkflowPhase
from src.common.timing import timed
from src.writing.causality.initializer import ensure_core_predicates
from .models import SceneCompletionCommand, SceneCompletionResult
import logging
from src.writing.memory_hierarchy import CompressedState
from src.writing.services.perception_propagation import PerceptionPropagationService

# 相变与吸引子系统导入
from src.writing.phase_transition import (
    PhaseTransitionDetector, PhaseTransitionHandler, PhaseTransition, PhaseTransitionType
)
from src.writing.attractor import NarrativeAttractorField, Attractor, AttractorType

logger = logging.getLogger(__name__)


class SceneCompletionService:
    """
    场景完成事务 - 一个完整的业务原子操作。
    未来拆分候选：SnapshotPersistence, ProgressUpdater, EventApplication
    """
    @staticmethod
    @timed("SceneCompletionService.execute")
    async def execute(cmd: SceneCompletionCommand) -> SceneCompletionResult:
        pool = get_db_pool()
        if not pool:
            return SceneCompletionResult(
                state_patch=StatePatch(error="Database pool unavailable"),
                error="No db pool"
            )

        if cmd.parsed_output is None:
            logger.error("SceneCompletionService: parsed_output is None")
            return SceneCompletionResult(
                state_patch=StatePatch(error="Missing parsed_output from validator", phase=WorkflowPhase.VALIDATING),
                error="Missing parsed_output"
            )

        new_world = WorldState.from_dict(cmd.current_world_state) if cmd.current_world_state else WorldState()
        events_applied = 0
        event_store = NarrativeEventStore(pool)

        # 用于相变检测的 compressed_state（将在合适时机构建）
        compressed_state_for_detection = None

        async with pool.acquire() as conn:
            async with conn.transaction():
                events_data = cmd.parsed_output.get("events", [])
                events = []
                for evt_dict in events_data:
                    evt_type = evt_dict.get("type")
                    if not evt_type:
                        logger.warning(f"Event missing 'type' field: {evt_dict}")
                        continue
                    
                    if evt_type == "discovery" and "importance" in evt_dict:
                        imp = evt_dict["importance"]
                        if isinstance(imp, int):
                            if imp >= 5:
                                evt_dict["importance"] = "critical"
                            elif imp >= 3:
                                evt_dict["importance"] = "high"
                            elif imp >= 1:
                                evt_dict["importance"] = "normal"
                            else:
                                evt_dict["importance"] = "low"
                        elif isinstance(imp, float):
                            if imp >= 5:
                                evt_dict["importance"] = "critical"
                            elif imp >= 3:
                                evt_dict["importance"] = "high"
                            elif imp >= 1:
                                evt_dict["importance"] = "normal"
                            else:
                                evt_dict["importance"] = "low"
                        elif isinstance(imp, bool):
                            evt_dict["importance"] = "critical" if imp else "low"
                    evt = event_from_dict(evt_type, evt_dict)
                    if evt:
                        events.append(evt)

                if events:
                    delta = StateDelta(events=events)
                    new_world = delta.apply_to(new_world)
                    events_applied = len(events)

                    # 保存原始事件 - 传入同一个 conn
                    for evt in events:
                        await event_store.append_event(
                            cmd.novel_id, evt, cmd.volume, cmd.chapter, cmd.scene_idx,
                            conn=conn  # 关键：复用事务连接
                        )

                    # 感知传播
                    world_before = WorldState.from_dict(cmd.current_world_state) if cmd.current_world_state else WorldState()
                    perception_events = await PerceptionPropagationService.propagate(
                        novel_id=cmd.novel_id,
                        volume=cmd.volume,
                        chapter=cmd.chapter,
                        scene_idx=cmd.scene_idx,
                        events=events,
                        world_state_before=world_before,
                    )
                    for pe in perception_events:
                        await event_store.append_event(
                            cmd.novel_id, pe, cmd.volume, cmd.chapter, cmd.scene_idx,
                            conn=conn  # 复用连接
                        )
                        delta_pe = StateDelta(events=[pe])
                        new_world = delta_pe.apply_to(new_world)
                        events_applied += 1

                    await ensure_core_predicates(cmd.novel_id, new_world)

                # ========== 相变检测与处理（放在事件应用之后、快照保存之前） ==========
                # 构建 compressed_state（用于相变检测）
                if cmd.character_intents or cmd.voice_memory:
                    compressed_state_for_detection = CompressedState(
                        volume_num=cmd.volume,
                        character_intents=cmd.character_intents or {},
                        voice_fingerprint=cmd.voice_memory or {},
                    )
                elif hasattr(cmd, 'compressed_state') and cmd.compressed_state:
                    compressed_state_for_detection = cmd.compressed_state
                
                # 加载已有相变
                existing_transitions = []
                if hasattr(new_world, 'phase_transitions') and new_world.phase_transitions:
                    existing_transitions = [
                        PhaseTransition.from_dict(pt) for pt in new_world.phase_transitions
                    ]
                
                # 检测新相变
                new_transitions = PhaseTransitionDetector.detect(
                    new_world,
                    compressed_state_for_detection,
                    existing_transitions,
                )
                
                # 应用相变
                for transition in new_transitions:
                    new_world = PhaseTransitionHandler.apply_transition(transition, new_world)
                    if not hasattr(new_world, 'phase_transitions'):
                        new_world.phase_transitions = []
                    new_world.phase_transitions.append(transition.to_dict())
                    logger.info(f"✅ Phase transition triggered: {transition.type.value}")
                
                # 如果触发了世界冲突相变，更新吸引子
                if any(t.type == PhaseTransitionType.WORLD_CONFLICT for t in new_transitions):
                    attractor_field = NarrativeAttractorField()
                    if hasattr(new_world, 'attractor_field') and new_world.attractor_field:
                        attractor_field = NarrativeAttractorField.from_dict(new_world.attractor_field)
                    attractor_field.register_attractor(Attractor(
                        id="world_conflict",
                        name="世界冲突",
                        type=AttractorType.CONFLICT,
                        weight=2.0,
                        decay_distance=20,
                    ))
                    new_world.attractor_field = attractor_field.to_dict()
                # ================================================================

                # 如果是章节最后一个场景，保存快照（包含可能更新的 phase_transitions 和 attractor_field）
                if cmd.scene_idx + 1 >= cmd.total_scenes:
                    last_id = await event_store.get_last_event_id(cmd.novel_id)
                    snap_mgr = SnapshotManager(pool)
                    # 构建最终的 compressed_state（优先使用已有，否则使用检测用的）
                    final_compressed = compressed_state_for_detection
                    if final_compressed is None and (cmd.character_intents or cmd.voice_memory):
                        final_compressed = CompressedState(
                            volume_num=cmd.volume,
                            character_intents=cmd.character_intents or {},
                            voice_fingerprint=cmd.voice_memory or {},
                        )
                    # 传入同一个 conn，确保快照与事件在同一个事务中
                    await snap_mgr.save_snapshot(
                        cmd.novel_id, new_world, last_id, cmd.volume, cmd.chapter,
                        compressed_state=final_compressed,
                        conn=conn
                    )

                # 更新 scene_execution_units 状态（这些操作不需要与事件在同一事务，可以独立，但已在事务内）
                await conn.execute(
                    """
                    UPDATE scene_execution_units
                    SET status = 'succeeded', completed_at = NOW(), updated_at = NOW()
                    WHERE novel_id = $1 AND volume_num = $2 AND chapter_num = $3 AND scene_index = $4
                    """,
                    cmd.novel_id, cmd.volume, cmd.chapter, cmd.scene_idx
                )

                # 更新 writing_progress
                new_scene_idx = cmd.scene_idx + 1
                await conn.execute(
                    """
                    INSERT INTO writing_progress (project_id, current_volume, current_chapter, current_scene, chapter_completed, last_updated)
                    VALUES ($1, $2, $3, $4, false, NOW())
                    ON CONFLICT (project_id) DO UPDATE SET
                        current_scene = EXCLUDED.current_scene,
                        last_updated = NOW()
                    """,
                    cmd.novel_id, cmd.volume, cmd.chapter, new_scene_idx
                )

        chapter_finished = (cmd.total_scenes > 0 and new_scene_idx >= cmd.total_scenes)
        patch = StatePatch(
            current_scene_index=new_scene_idx,
            current_state=new_world.to_dict(),
            retry_count=0,
            validation_result=cmd.parsed_output,
        )
        if chapter_finished:
            patch.phase = WorkflowPhase.TRANSITIONING
        else:
            patch.phase = WorkflowPhase.WRITING

        return SceneCompletionResult(
            state_patch=patch,
            chapter_finished=chapter_finished,
            events_applied=events_applied,
        )