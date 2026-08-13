# src/writing/services/scene_completion.py
import asyncpg
from pathlib import Path
from src.db import get_db_pool
from src.writing.event_store import NarrativeEventStore
from src.writing.snapshot_manager import SnapshotManager
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
                    
                # ========== Phase 13.2.1: Projection 更新（事务内） ==========
                if cmd.narrative_intent and events:
                    try:
                        from src.writing.projection_service import NarrativeProjectionService
                        from src.writing.projection_updater import ProjectionUpdater
                        from src.writing.exceptions import ProjectionUpdateFailed
                        import inspect

                        # 关键：传入同一个 conn，确保事务一致
                        service = NarrativeProjectionService(conn=conn)
                        current_result = service.load_current()
                        current = await current_result if inspect.isawaitable(current_result) else current_result

                        updater = ProjectionUpdater()
                        new_projection = updater.update(
                            previous=current,
                            intent=cmd.narrative_intent,
                            events=events
                        )
                        save_result = service.save(new_projection)
                        if inspect.isawaitable(save_result):
                            await save_result

                        logger.info(f"✅ Projection 更新成功 (version {new_projection.version})")
                    except Exception as e:
                        logger.error(f"Projection 更新失败，事务回滚: {e}", exc_info=True)
                        raise ProjectionUpdateFailed("Projection update failed") from e
                # ===========================================================

                # ========== 保存场景正文到文件（独立于事件） ==========
                logger.info(f"[SAVE_DEBUG] cmd.parsed_output keys: {list(cmd.parsed_output.keys()) if cmd.parsed_output else 'None'}")
                scene_text_to_save = None
                if cmd.parsed_output:
                    scene_text_to_save = cmd.parsed_output.get("scene_text")
                    logger.info(f"[SAVE_DEBUG] scene_text from parsed_output length = {len(scene_text_to_save) if scene_text_to_save else 0}")

                if not scene_text_to_save and hasattr(cmd, 'raw_output') and cmd.raw_output:
                    import re
                    match = re.search(r'"scene_text"\s*:\s*"((?:[^"\\]|\\.)*)"', cmd.raw_output, re.DOTALL)
                    if match:
                        scene_text_to_save = match.group(1).replace('\\"', '"').replace('\\n', '\n')
                        logger.warning(f"[SAVE_DEBUG] Extracted scene_text from raw_output, length={len(scene_text_to_save)}")

                if scene_text_to_save and len(scene_text_to_save.strip()) >= 10:
                    await SceneCompletionService._save_scene_to_file(cmd, scene_text_to_save)
                else:
                    logger.error(f"[SAVE_DEBUG] Cannot save scene: scene_text missing or too short (len={len(scene_text_to_save) if scene_text_to_save else 0})")
                    # 可选：保存调试信息
                    try:
                        debug_dir = Path(f"data/novels/{cmd.novel_id}/debug")
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        debug_file = debug_dir / f"vol_{cmd.volume:03d}_chap_{cmd.chapter:03d}_scene_{cmd.scene_idx:02d}_raw.txt"
                        with open(debug_file, "w", encoding="utf-8") as f:
                            f.write(str(cmd.parsed_output) + "\n\n" + str(getattr(cmd, 'raw_output', '')))
                        logger.info(f"[SAVE_DEBUG] Saved raw output to {debug_file} for debugging")
                    except Exception as e:
                        logger.error(f"[SAVE_DEBUG] Failed to save debug output: {e}")
                # =========================================================
                    
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

    @staticmethod
    async def _save_scene_to_file(cmd: SceneCompletionCommand, scene_text: str):
        """保存场景正文到章节文件"""
        logger.info(f"_save_scene_to_file called for chapter {cmd.chapter}, scene {cmd.scene_idx}, text length={len(scene_text)}")
        if not scene_text or len(scene_text.strip()) < 50:
            logger.warning(f"Scene text too short ({len(scene_text)} chars), skip saving")
            return
        try:
            novel_data_dir = Path(f"data/novels/{cmd.novel_id}")
            volumes_dir = novel_data_dir / f"vol_{cmd.volume:03d}"
            volumes_dir.mkdir(parents=True, exist_ok=True)
            chapter_file = volumes_dir / f"chap_{cmd.chapter:03d}.txt"
            mode = "a" if chapter_file.exists() else "w"
            with open(chapter_file, mode, encoding="utf-8") as f:
                if mode == "a":
                    f.write("\n\n<!-- scene break -->\n\n")
                f.write(scene_text.strip())
            logger.info(f"✅ Saved scene to {chapter_file} (mode={mode}, length={len(scene_text)})")
        except Exception as e:
            logger.error(f"Failed to save scene: {e}", exc_info=True)