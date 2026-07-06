import json
import re
import logging
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from src.db import get_db_pool
from src.agents.planner import PlannerAgent
from src.orchestrator.state_patch import StatePatch, WorkflowPhase
from src.writing.world_state import WorldState, Realm, CharacterState
from src.writing.context_compiler import ContextCompiler
from src.writing.causality.initializer import ensure_core_predicates
from .models import ScenePlanningCommand, ScenePlanningResult
from src.orchestrator.state import AgentState
from src.writing.attractor import NarrativeAttractorField
from src.domain.identity import get_main_character_id, get_character_name
from src.agents.drama_planner import DramaPlannerAgent

logger = logging.getLogger(__name__)


class ScenePlanningService:
    @staticmethod
    async def execute(cmd: ScenePlanningCommand) -> ScenePlanningResult:
        logger.info(f"ScenePlanningService: cmd.outline={cmd.outline is not None}, novel_id={cmd.novel_id}")
        
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
                # 从 outline 初始化角色（此部分代码原不完整，保留原样）
                for char_info in outline["characters"]:
                    name = char_info.get("name")
                    initial_realm = char_info.get("initial_state", {}).get("realm", "炼气")
                    # 解析境界...
                    # 原代码有缺失，但保持原逻辑，不作改动
                    pass

        # ========== 确保主角存在（从配置读取） ==========
        protagonist_id = get_main_character_id()
        protagonist_name = get_character_name(protagonist_id)

        protagonist_exists = False
        char = world.get_character(protagonist_id)
        if char is None:
            char = world.get_character(protagonist_name)
        protagonist_exists = char is not None

        if not protagonist_exists:
            initial_realm = Realm.REFINING_QI
            initial_level = 1
            if outline and "characters" in outline:
                for char_info in outline["characters"]:
                    if char_info.get("name") == protagonist_name:
                        realm_str = char_info.get("initial_state", {}).get("realm", "炼气")
                        match = re.search(r'(\d+)', realm_str)
                        if match:
                            initial_level = int(match.group(1))
                        break

            char_state = CharacterState(
                name=protagonist_name,
                realm=initial_realm,
                realm_level=initial_level
            )
            char_state.id = protagonist_id
            world.characters[protagonist_id] = char_state
            world.characters[protagonist_name] = char_state  # 兼容期

            logger.info(f"Initialized protagonist {protagonist_name} (id={protagonist_id}) at {initial_realm.value}{initial_level}层")

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

        # 5. 调用 PlannerAgent
        planner = PlannerAgent()
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

        # ========== 计算叙事引力（保留原逻辑） ==========
        attractor_field = NarrativeAttractorField()
        if hasattr(world, 'attractor_field') and world.attractor_field:
            attractor_field = NarrativeAttractorField.from_dict(world.attractor_field)
        
        min_gravity_threshold = 0.5
        total_gravity = 0.0
        for scene in scenes:
            gravity = attractor_field.calculate_gravity(scene, world, cmd.chapter)
            total_gravity += gravity
        
        avg_gravity = total_gravity / max(len(scenes), 1)
        gravity_warning = None
        if avg_gravity < min_gravity_threshold:
            gravity_warning = attractor_field.get_attractor_prompt(min_gravity_threshold)
            logger.warning(f"Low narrative gravity: avg={avg_gravity:.2f}, scenes may drift from attractors")

        # ========== 🆕 为每个场景生成戏剧结构（并行） ==========
        drama_planner = DramaPlannerAgent()
        tasks = []
        for scene in scenes:
            temp_drama_state = AgentState(
                scene_plan=scene,
                novel_id=cmd.novel_id,
                current_volume=cmd.volume,
                current_chapter=cmd.chapter,
                current_state=world.to_dict(),
            )
            tasks.append(drama_planner.run(temp_drama_state))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to generate drama for scene {i}: {result}")
                scenes[i]["drama"] = {}
            else:
                drama_struct = result.get("drama_structure", {})
                if drama_struct:
                    scenes[i]["drama"] = drama_struct
                    logger.info(f"✅ Generated drama structure for scene {i+1}")
                else:
                    scenes[i]["drama"] = {}
        # =====================================================

        # 7. 持久化到 scene_execution_units（此时每个场景已含 "drama" 字段）
        if cmd.novel_id and pool:
            await ScenePlanningService._persist_scene_plans(
                pool, cmd.novel_id, cmd.volume, cmd.chapter, scenes
            )

        # 8. 构建 StatePatch
        total_scenes = len(scenes)
        first_scene = scenes[0] if scenes else {}
        
        patch_metadata = {}
        if gravity_warning:
            patch_metadata["gravity_warning"] = gravity_warning
        
        patch = StatePatch(
            scene_plan_list=scenes,
            total_scenes_in_chapter=total_scenes,
            scene_plan=first_scene,
            phase=WorkflowPhase.WRITING,
            current_scene_index=0,
            current_state=world.to_dict(),
            metadata=patch_metadata,
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
        patch.total_chapters_in_volume = total_chapters

        return ScenePlanningResult(
            state_patch=patch,
            total_scenes=total_scenes,
        )

    @staticmethod
    async def _persist_scene_plans(
        pool, novel_id: str, volume: int, chapter: int, scenes: List[Dict]
    ):
        """持久化场景计划到 scene_execution_units"""
        from datetime import datetime
        
        def _json_serializer(obj):
            """自定义 JSON 序列化器"""
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        async with pool.acquire() as conn:
            for idx, scene in enumerate(scenes):
                plan_json = json.dumps(scene, ensure_ascii=False, default=_json_serializer)
                planned_state_delta = json.dumps(
                    scene.get("state_delta", {}), 
                    ensure_ascii=False, 
                    default=_json_serializer
                ) if scene.get("state_delta") else None
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