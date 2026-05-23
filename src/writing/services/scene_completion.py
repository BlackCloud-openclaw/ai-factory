# src/writing/services/scene_completion.py
import asyncpg
from src.db import get_db_pool
from src.writing.event_store import NarrativeEventStore
from src.writing.snapshot import SnapshotManager
from src.writing.delta import StateDelta
from src.writing.world_state import WorldState
from src.writing.events import event_from_dict
from src.orchestrator.state_patch import StatePatch, WorkflowPhase
from .models import SceneCompletionCommand, SceneCompletionResult
import logging

logger = logging.getLogger(__name__)


class SceneCompletionService:
    """
    场景完成事务 - 一个完整的业务原子操作。
    未来拆分候选：SnapshotPersistence, ProgressUpdater, EventApplication
    """
    @staticmethod
    async def execute(cmd: SceneCompletionCommand) -> SceneCompletionResult:
        pool = get_db_pool()
        if not pool:
            return SceneCompletionResult(
                state_patch=StatePatch(error="Database pool unavailable"),
                error="No db pool"
            )

        new_world = WorldState.from_dict(cmd.current_world_state) if cmd.current_world_state else WorldState()
        events_applied = 0

        async with pool.acquire() as conn:
            async with conn.transaction():
                # 1. 应用事件
                events_data = cmd.parsed_output.get("events", [])
                events = []
                for evt_dict in events_data:
                    # ----- 修复 discovery.importance 类型 -----
                    if evt_dict.get("type") == "discovery" and "importance" in evt_dict:
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
                            evt_dict["importance"] = "critical" if imp >= 5 else "high" if imp >= 3 else "normal" if imp >= 1 else "low"
                        elif isinstance(imp, bool):
                            evt_dict["importance"] = "critical" if imp else "low"
                    # ---------------------------------------
                    evt_type = evt_dict.get("type")
                    if evt_type:
                        evt = event_from_dict(evt_type, evt_dict)
                        if evt:
                            events.append(evt)

                if events:
                    delta = StateDelta(events=events)
                    new_world = delta.apply_to(new_world)
                    events_applied = len(events)

                    event_store = NarrativeEventStore(pool)
                    for evt in events:
                        await event_store.append_event(
                            cmd.novel_id, evt, cmd.volume, cmd.chapter, cmd.scene_idx
                        )

                    # 最后一个场景保存快照
                    if cmd.scene_idx + 1 >= cmd.total_scenes:
                        last_id = await event_store.get_last_event_id(cmd.novel_id)
                        snap_mgr = SnapshotManager(pool)
                        await snap_mgr.save_snapshot(
                            cmd.novel_id, new_world, last_id, cmd.volume, cmd.chapter
                        )

                # 2. 更新 scene_execution_units 状态为 succeeded
                await conn.execute(
                    """
                    UPDATE scene_execution_units
                    SET status = 'succeeded', completed_at = NOW(), updated_at = NOW()
                    WHERE novel_id = $1 AND volume_num = $2 AND chapter_num = $3 AND scene_index = $4
                    """,
                    cmd.novel_id, cmd.volume, cmd.chapter, cmd.scene_idx
                )

                # 3. 更新 writing_progress
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

        # 4. 构建 StatePatch
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