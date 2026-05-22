import json
import re
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.db import get_db_pool
from src.agents.planner import PlannerAgent
from src.orchestrator.state_patch import StatePatch, WorkflowPhase
from src.writing.world_state import WorldState, Realm, CharacterState
from src.writing.context_compiler import ContextCompiler
from src.writing.causality.initializer import ensure_core_predicates
from .models import ScenePlanningCommand, ScenePlanningResult
from src.orchestrator.state import AgentState

logger = logging.getLogger(__name__)


class ScenePlanningService:
    @staticmethod
    async def execute(cmd: ScenePlanningCommand) -> ScenePlanningResult:
        pool = get_db_pool()
        updates = {}

        # 1. 重新加载 outline（如果需要）
        outline = cmd.outline
        if outline is None and cmd.novel_id and pool:
            try:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT outline FROM novels WHERE novel_id = $1", cmd.novel_id
                    )
                    if row and row["outline"]:
                        outline = json.loads(row["outline"])
                        logger.info(f"✅ Reloaded outline for {cmd.novel_id}")
            except Exception as e:
                logger.error(f"Failed to reload outline: {e}")

        # 2. 获取或初始化世界状态
        world = None
        if cmd.current_state:
            world = WorldState.from_dict(cmd.current_state)
        else:
            world = WorldState()
            if outline and "characters" in outline:
                for char_info in outline["characters"]:
                    name = char_info.get("name")
                    initial_realm = char_info.get("initial_state", {}).get("realm", "炼气")
                    # 解析境界和层级
                    level = 1
                    realm_str = initial_realm
                    match = re.search(r'(\d+)', realm_str)
                    if match:
                        level = int(match.group(1))
                        realm_str = realm_str.replace(str(level), "").strip()
                    realm_map = {"炼气": Realm.REFINING_QI, "筑基": Realm.FOUNDATION, "金丹": Realm.GOLDEN_CORE}
                    realm_enum = realm_map.get(realm_str, Realm.REFINING_QI)
                    char_state = CharacterState(name=name, realm=realm_enum, realm_level=level)
                    world.characters[name] = char_state

        # 3. 确保核心谓词已投影
        if cmd.novel_id:
            await ensure_core_predicates(cmd.novel_id, world)

        # 4. 编译上下文（用于 planner prompt）
        compiler = ContextCompiler()
        current_volume_outline = None
        if outline and "volumes" in outline:
            volumes = outline.get("volumes", [])
            vol_idx = cmd.volume - 1
            if 0 <= vol_idx < len(volumes):
                current_volume_outline = volumes[vol_idx]
        compiled = compiler.compile_for_planner(
            world, cmd.volume, cmd.chapter,
            current_volume_outline or outline or {}
        )
        # 注意：compiled 将被注入到 PlannerAgent 的 metadata 中
        # 由于我们无法直接修改 cmd 对象，我们需要通过临时状态传递给 planner
        # 更好的方式：让 PlannerAgent 从 state.metadata 读取，我们在调用前临时构造一个 state
        # 但为了保持服务独立性，我们可以在调用 PlannerAgent 之前构造一个临时 AgentState

        # 5. 调用 PlannerAgent
        planner = PlannerAgent()
        # 构造临时 AgentState 以传递必要信息
        temp_state = AgentState(
            novel_id=cmd.novel_id,
            task_type=cmd.task_type,
            user_input=cmd.user_input,
            outline=outline,
            current_volume=cmd.volume,
            current_chapter=cmd.chapter,
            current_state=world.to_dict(),
            metadata={"compiled_context": compiled},
        )
        planner_updates = await planner.run(temp_state)
        scene_plan_data = planner_updates.get("scene_plan")
        if not scene_plan_data:
            logger.warning("ScenePlanningService: no scene_plan returned from planner")
            return ScenePlanningResult(
                state_patch=StatePatch(error="No scene plan generated"),
                error="No scene plan generated"
            )

        # 6. 标准化场景计划
        if isinstance(scene_plan_data, dict):
            scenes = scene_plan_data.get("scenes", [])
        elif isinstance(scene_plan_data, list):
            scenes = scene_plan_data
        else:
            scenes = []

        if not scenes:
            logger.warning("ScenePlanningService: empty scenes list")
            return ScenePlanningResult(
                state_patch=StatePatch(error="Empty scenes list"),
                error="Empty scenes list"
            )

        for i, scene in enumerate(scenes):
            if "must_events" not in scene or not scene["must_events"]:
                scene["must_events"] = [f"推进主线剧情（场景{i+1}）"]
            else:
                scene["must_events"] = [e for e in scene["must_events"] if "推进主线剧情" not in e]
            if "state_delta" not in scene:
                scene["state_delta"] = {"events": []}
            if "depends_on" not in scene:
                scene["depends_on"] = []
            if "scene_id" not in scene:
                scene["scene_id"] = i + 1

        # 7. 持久化到 scene_execution_units
        if cmd.novel_id and pool:
            await ScenePlanningService._persist_scene_plans(
                pool, cmd.novel_id, cmd.volume, cmd.chapter, scenes
            )

        # 8. 构建 StatePatch
        total_scenes = len(scenes)
        first_scene = scenes[0] if scenes else {}
        patch = StatePatch(
            scene_plan_list=scenes,
            total_scenes_in_chapter=total_scenes,
            scene_plan=first_scene,
            phase=WorkflowPhase.WRITING,  # 计划生成后进入写作阶段
            current_scene_index=0,        # 重置场景索引
        )

        # 9. 确保 total_chapters_in_volume
        total_chapters = cmd.total_chapters_in_volume
        if total_chapters == 0 and outline and "volumes" in outline:
            volumes = outline["volumes"]
            vol_idx = cmd.volume - 1
            if 0 <= vol_idx < len(volumes):
                total_chapters = len(volumes[vol_idx].get("chapters", []))
        if total_chapters == 0:
            total_chapters = 10
            logger.warning(f"Could not determine total_chapters_in_volume, using default {total_chapters}")
        patch.total_chapters_in_volume = total_chapters  # 注意：该字段需要在 StatePatch 中添加

        return ScenePlanningResult(
            state_patch=patch,
            total_scenes=total_scenes,
        )

    @staticmethod
    async def _persist_scene_plans(
        pool, novel_id: str, volume: int, chapter: int, scenes: List[Dict]
    ):
        """持久化场景计划到 scene_execution_units（复用原有辅助函数）"""
        # 复用 nodes.py 中的 _persist_scene_plans 函数，为避免循环导入，这里重新实现简单版
        from src.db import get_db_pool
        async with pool.acquire() as conn:
            for idx, scene in enumerate(scenes):
                plan_json = json.dumps(scene, ensure_ascii=False)
                planned_state_delta = json.dumps(scene.get("state_delta", {}), ensure_ascii=False) if scene.get("state_delta") else None
                await conn.execute("""
                    INSERT INTO scene_execution_units 
                    (novel_id, volume_num, chapter_num, scene_index, status, plan_json, planned_state_delta, retry_count, max_retries, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, 'pending', $5, $6, 0, 2, NOW(), NOW())
                    ON CONFLICT (novel_id, volume_num, chapter_num, scene_index)
                    DO UPDATE SET
                        plan_json = EXCLUDED.plan_json,
                        planned_state_delta = EXCLUDED.planned_state_delta,
                        status = 'pending',
                        retry_count = 0,
                        updated_at = NOW()
                """, novel_id, volume, chapter, idx, plan_json, planned_state_delta)
        logger.info(f"Persisted {len(scenes)} scenes to scene_execution_units for chapter {chapter}")