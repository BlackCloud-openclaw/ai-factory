# src/writing/services/scene_planning.py
"""
场景计划生成服务 - 核心业务事务

职责：
1. 加载 Outline 和世界状态
2. 调用 PlannerAgent 生成场景计划
3. 为每个场景生成戏剧结构 (DramaPlannerAgent)
4. Phase 13.2.3A: 标准化 PlanningContract (ContractNormalizer)
5. Phase 14.0A-1: 验证场景事件结构 (SceneEventValidator)
6. 持久化场景计划到 scene_execution_units
7. 构建 StatePatch 返回
"""

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
from src.writing.attractor import NarrativeAttractorField
from src.domain.identity import get_main_character_id, get_character_name
from src.agents.drama_planner import DramaPlannerAgent
from src.narrative.intent import IntentResolver
from src.narrative.adaptive import create_adaptive_resolver_with_rollout
from src.config import config
from src.orchestrator.state import AgentState

# Phase 13.2.3A: Contract Normalizer
from src.writing.contract_normalizer import ContractNormalizer
from src.writing.planning_contract import PlanningContract

# Phase 14.0A-1: Scene Event Validator
from src.writing.scene_event_validator import (
    SceneEventValidator,
    SceneEventRequirement,
    EventValidationStatus,
)

# Phase 14.0B-2: Contract Validation Exceptions (with alias to avoid UnboundLocalError)
from src.writing.contracts.exceptions import (
    InvalidSceneContract,
    ContractValidationError as ContractErr,
)

from .models import ScenePlanningCommand, ScenePlanningResult
from src.writing.contract_validator import ContractConsistencyValidator


logger = logging.getLogger(__name__)


class ScenePlanningService:
    """
    场景计划生成服务。

    Phase 13.2.3A 扩展：
    - 使用 ContractNormalizer 标准化 PlanningContract
    - Normalizer 为类级别单例，共享审计日志

    Phase 14.0A-1 扩展：
    - 使用 SceneEventValidator 验证 must_events 结构
    - INVALID 事件触发 InvalidSceneContract 异常
    - WARNING 事件记录日志，允许继续
    """

    # Phase 13.2.3A: Normalizer 类级别单例
    _normalizer: Optional[ContractNormalizer] = None

    @classmethod
    def _get_normalizer(cls) -> ContractNormalizer:
        """获取 Normalizer 单例（懒加载）。"""
        if cls._normalizer is None:
            cls._normalizer = ContractNormalizer()
            logger.info("ContractNormalizer singleton initialized")
        return cls._normalizer

    @staticmethod
    async def execute(cmd: ScenePlanningCommand) -> ScenePlanningResult:
        """
        执行场景计划生成事务。

        Args:
            cmd: 场景计划生成命令

        Returns:
            ScenePlanningResult: 包含 StatePatch 和场景计划列表
        """
        logger.info(f"ScenePlanningService: cmd.outline={cmd.outline is not None}, novel_id={cmd.novel_id}")

        # ---------- Phase 9.4：处理 IntentResolver ----------
        if cmd.intent_resolver is not None:
            resolver = cmd.intent_resolver
            logger.debug("Using injected IntentResolver")
        else:
            if config.adaptive_runtime_enabled:
                conflict_resolver = create_adaptive_resolver_with_rollout(
                    rollout_percentage=config.adaptive_rollout_percentage,
                    enable_telemetry=True,
                    novel_id=cmd.novel_id,
                    chapter=cmd.chapter,
                    scene=0,
                )
                resolver = IntentResolver(conflict_resolver=conflict_resolver)
                logger.info(f"Adaptive runtime enabled (service), rollout={config.adaptive_rollout_percentage}%")
            else:
                resolver = IntentResolver()
                logger.info("Adaptive runtime disabled (service), using rule selector")

        if cmd.metadata is None:
            cmd.metadata = {}
        cmd.metadata["intent_resolver"] = resolver

        # ---------- 1. 加载 Outline ----------
        pool = get_db_pool()
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

        # ---------- 2. 初始化世界状态 ----------
        world = None
        if cmd.current_state:
            world = WorldState.from_dict(cmd.current_state)
        else:
            world = WorldState()
            if outline and "characters" in outline:
                for char_info in outline["characters"]:
                    name = char_info.get("name")
                    initial_realm = char_info.get("initial_state", {}).get("realm", "炼气")
                    # 简化：只做基础初始化

        # ---------- 3. 确保主角存在 ----------
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
            world.characters[protagonist_name] = char_state
            logger.info(f"Initialized protagonist {protagonist_name} (id={protagonist_id}) at {initial_realm.value}{initial_level}层")

        # ---------- 4. 确保核心谓词 ----------
        if cmd.novel_id:
            await ensure_core_predicates(cmd.novel_id, world)

        # ---------- 5. 编译上下文 ----------
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

        # ---------- 6. 调用 PlannerAgent ----------
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

        # ================================================================
        # ⭐ Phase 13.2.2 关键修改：从 temp_state.metadata 捕获 planner_outputs
        # ================================================================
        planner_outputs = planner_updates.get("planner_outputs", [])
        # ================================================================

        scene_plan_data = planner_updates.get("scene_plan")
        if not scene_plan_data:
            logger.warning("ScenePlanningService: no scene_plan returned from planner")
            return ScenePlanningResult(
                state_patch=StatePatch(error="No scene plan generated"),
                error="No scene plan generated"
            )

        # ---------- 7. 标准化场景计划 ----------
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
            if "must_events" not in scene or not isinstance(scene.get("must_events"), list):
                scene["must_events"] = []
            else:
                scene["must_events"] = [e for e in scene["must_events"] if "推进主线剧情" not in e]
            if "state_delta" not in scene:
                scene["state_delta"] = {"events": []}
            if "depends_on" not in scene:
                scene["depends_on"] = []
            if "scene_id" not in scene:
                scene["scene_id"] = i + 1

        # ================================================================
        # ⭐ Phase 14.0A-1: 场景事件验证（P1 降级版）
        # ================================================================
        logger.info(f"开始场景事件验证: {len(scenes)} 个场景")
        patch_metadata = {}

        requirement = SceneEventRequirement()
        validation_results = []
        blocking_errors = 0

        for idx, scene in enumerate(scenes):
            events = scene.get("must_events", [])
            result = SceneEventValidator.validate_scene(events, requirement)
            validation_results.append(result)

            # ========== P1 修改：只阻断真正的占位符事件 ==========
            if not result.valid:
                invalid_events = SceneEventValidator.get_invalid_events(result)
                
                # 检查是否有真正的阻塞错误（占位符事件）
                # 占位符特征：包含“推进”、“主线”、“剧情”、“场景X”等
                has_placeholder = any(
                    "推进" in e or "主线" in e or "剧情" in e or re.search(r'场景\d+', e)
                    for e in invalid_events
                )
                
                if has_placeholder or blocking_errors > 0:
                    blocking_errors += result.invalid_count
                    # 有占位符或结构错误 → 阻断
                    raise InvalidSceneContract(
                        scene_index=idx,
                        invalid_events=invalid_events,
                        validation_result=result,
                    )
                else:
                    # 只有语义模糊但不阻塞的事件 → 记录警告，允许继续
                    logger.info(f"Scene {idx} event validation warnings (non-blocking): {result.summary}")
                    if "scene_event_warnings" not in patch_metadata:
                        patch_metadata["scene_event_warnings"] = []
                    patch_metadata["scene_event_warnings"].append({
                        "scene_index": idx,
                        "summary": result.summary,
                        "invalid_events": invalid_events,
                    })
            # ============================================================

        # 记录验证摘要到 metadata（用于后续审计）
        patch_metadata["scene_event_validation"] = {
            "total": len(scenes),
            "valid": sum(1 for r in validation_results if r.valid),
            "invalid": sum(1 for r in validation_results if r.contract_quality == "invalid"),
            "blocking_errors": blocking_errors,
            "contract_qualities": [r.contract_quality for r in validation_results],
        }

        logger.info(
            f"✅ 场景事件验证完成: "
            f"{patch_metadata['scene_event_validation']['valid']}/{len(scenes)} 通过, "
            f"{blocking_errors} 个阻塞错误"
        )
        
        # ========== 在这里插入调试日志 ==========
        if scenes:
            logger.info(f"🔍 Scene 0 keys in scene_planning: {list(scenes[0].keys())}")
            logger.info(f"🔍 Scene 0 planning_contract in scene_planning: {scenes[0].get('planning_contract', 'MISSING')}")
        else:
            logger.warning("🔍 scenes is empty in scene_planning")

        # ================================================================
        # ⭐ Phase 14.0B-2: 检查原始 Contract 是否包含 state_changes
        # ================================================================
        logger.info("=== 开始检查原始 Contract 的 state_changes ===")
        logger.info("🔥🔥🔥 USING CHECK CODE v4.0 🔥🔥🔥")
        for idx, scene in enumerate(scenes):
            contract_data = scene.get("planning_contract")
            if contract_data is None:
                logger.warning(f"Scene {idx}: planning_contract is None, skipping state_changes check")
                # 这里直接跳过，但因为我们修了 planner.py，理论上不会发生
                continue
            observables = contract_data.get("observables", {})            
            state_changes = observables.get("state_changes", [])
            state_changes_count = len(state_changes)
            logger.info(f"Scene {idx}: state_changes count = {state_changes_count} (from planning_contract)")
            
            if state_changes_count == 0:
                scene_id = contract_data.get("scene_id", f"scene_{idx}")
                logger.error(f"Scene {idx} missing state_changes in planning_contract.observables")
                raise ContractErr(
                    scene_id=scene_id,
                    errors=["原始 Planning Contract 缺少 state_changes，Planner 必须提供可验证状态变化"],
                    warnings=[]
                )
            logger.info(f"Scene {idx}: state_changes found ({state_changes_count} changes)")
        logger.info("=== 原始 Contract state_changes 检查完成 ===")
        # ================================================================

        # ================================================================
        # ⭐ Phase 13.2.3A: Contract Normalization
        # ================================================================
        normalizer = ScenePlanningService._get_normalizer()
        normalized_count = 0

        for idx, scene in enumerate(scenes):
            contract_data = scene.get("planning_contract")
            if contract_data:
                try:
                    # 解析为 PlanningContract 对象
                    contract = PlanningContract(**contract_data)
                    # 标准化
                    normalized_contract = normalizer.normalize(contract)
                    # 回写
                    scene["planning_contract"] = normalized_contract.model_dump()
                    normalized_count += 1
                    logger.debug(
                        f"Normalized contract for scene {idx}: "
                        f"enriched={normalized_contract.enrichment.enriched}, "
                        f"rules={normalized_contract.enrichment.rules_applied}, "
                        f"state_changes={len(normalized_contract.observables.state_changes)}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to normalize contract for scene {idx}: {e}")
                    # 保留原 contract，不阻断流程

        if normalized_count > 0:
            logger.info(f"✅ Normalized {normalized_count} planning contracts (Phase 13.2.3A)")
        # ================================================================

        # ---------- 8. 叙事引力 ----------
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
            logger.warning(f"Low narrative gravity: avg={avg_gravity:.2f}")

        # ---------- 9. 生成戏剧结构 ----------
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

        # ---------- 10. 持久化 ----------
        if cmd.novel_id and pool:
            await ScenePlanningService._persist_scene_plans(
                pool, cmd.novel_id, cmd.volume, cmd.chapter, scenes
            )

        # ---------- 11. 构建 StatePatch ----------
        total_scenes = len(scenes)
        first_scene = scenes[0] if scenes else {}

        # 合并 metadata（包含验证结果）
        if gravity_warning:
            patch_metadata["gravity_warning"] = gravity_warning

        # Phase 13.2.3A: 在 metadata 中记录 Normalizer 审计摘要
        if normalized_count > 0:
            audit_logs = normalizer.get_audit_logs()
            if audit_logs:
                # 取最后一条审计记录（对应本次执行）
                last_audit = audit_logs[-1]
                patch_metadata["contract_normalizer_audit"] = {
                    "contract_id": last_audit.get("contract_id"),
                    "signals": last_audit.get("signals"),
                    "rules_applied": last_audit.get("rules_applied"),
                    "enrichment_applied": last_audit.get("enrichment_applied"),
                }

        patch = StatePatch(
            scene_plan_list=scenes,
            total_scenes_in_chapter=total_scenes,
            scene_plan=first_scene,
            phase=WorkflowPhase.WRITING,
            current_scene_index=0,
            current_state=world.to_dict(),
            metadata=patch_metadata,
        )

        # ================================================================
        # ⭐ Phase 14.0B-2: Contract 一致性验证
        # ================================================================
        contract_validation_results = []
        contract_validation_errors = 0

        # ================================================================
        # Phase 14.0C-3C: 自动修复重复的 StateChange ID
        # ================================================================
        import hashlib
        for scene in scenes:
            contract_data = scene.get("planning_contract")
            if not contract_data:
                continue
            state_changes = contract_data.get("observables", {}).get("state_changes", [])
            if not state_changes:
                continue

            scene_id = contract_data.get("scene_id", "unknown")
            existing_ids = set()
            modified = False

            for sc in state_changes:
                sc_id = sc.get("id", "")
                if sc_id and sc_id not in existing_ids:
                    existing_ids.add(sc_id)
                else:
                    # 生成新 ID（基于场景 ID + 类型 + 关键字段的哈希）
                    content_str = f"{scene_id}|{sc.get('type')}|{sc.get('from_char')}|{sc.get('to_char')}|{sc.get('actor')}|{sc.get('item')}|{sc.get('name')}"
                    new_id = hashlib.sha256(content_str.encode()).hexdigest()[:12]
                    # 如果新 ID 仍然冲突，添加后缀
                    suffix = 0
                    while new_id in existing_ids:
                        suffix += 1
                        new_id = hashlib.sha256((content_str + str(suffix)).encode()).hexdigest()[:12]
                    sc["id"] = new_id
                    existing_ids.add(new_id)
                    modified = True
                    logger.warning(f"Fixed duplicate StateChange ID for scene {scene_id}, new ID={new_id}")

            if modified:
                # 更新 contract_data 中的 observables
                contract_data["observables"]["state_changes"] = state_changes
                scene["planning_contract"] = contract_data
                logger.info(f"Fixed {len(state_changes)} StateChange IDs for scene {scene_id}")

        # ================================================================
        # ⭐ Phase 14.0B-2: Contract 一致性验证（原有代码）
        # ================================================================

        for idx, scene in enumerate(scenes):
            contract_data = scene.get("planning_contract")
            if contract_data:
                try:
                    contract = PlanningContract(**contract_data)
                    result = ContractConsistencyValidator.validate(contract)
                    contract_validation_results.append(result)

                    if not result.valid:
                        contract_validation_errors += 1
                        logger.error(
                            f"Scene {idx} contract validation failed: {result.errors}"
                        )
                        # 使用增强异常，包含完整错误列表
                        raise ContractErr.from_validation_result(
                            contract.scene_id,
                            result
                        )

                    if result.warnings:
                        for warning in result.warnings:
                            logger.warning(f"Scene {idx} contract warning: {warning}")

                except ContractErr:
                    raise
                except Exception as e:
                    logger.error(f"Scene {idx} contract validation error: {e}")
                    raise ContractErr(
                        scene_id=scene.get("scene_id", f"scene_{idx}"),
                        errors=[f"Unexpected validation error: {str(e)}"],
                        warnings=[],
                    )

        # 记录验证摘要到 metadata
        patch_metadata["contract_validation"] = {
            "total": len(scenes),
            "valid": sum(1 for r in contract_validation_results if r.valid),
            "invalid": contract_validation_errors,
            "warnings": sum(1 for r in contract_validation_results if r.warnings),
        }

        logger.info(
            f"✅ Contract validation completed: "
            f"{patch_metadata['contract_validation']['valid']}/{len(scenes)} valid, "
            f"{patch_metadata['contract_validation']['warnings']} warnings"
        )
        # ================================================================

        # ---------- 12. 确定总章节数 ----------
        total_chapters = cmd.total_chapters_in_volume
        if total_chapters == 0 and outline and "volumes" in outline:
            volumes = outline["volumes"]
            vol_idx = cmd.volume - 1
            if 0 <= vol_idx < len(volumes):
                total_chapters = len(volumes[vol_idx].get("chapters", []))
        if total_chapters == 0:
            total_chapters = 10
            logger.warning(f"Using default total_chapters: {total_chapters}")
        patch.total_chapters_in_volume = total_chapters

        # ================================================================
        # ⭐ Phase 13.2.2 关键修改：返回 planner_outputs
        # ================================================================
        return ScenePlanningResult(
            state_patch=patch,
            total_scenes=total_scenes,
            planner_outputs=planner_outputs,   # 显式契约传递
        )
        # ================================================================

    @staticmethod
    async def _persist_scene_plans(
        pool, novel_id: str, volume: int, chapter: int, scenes: List[Dict]
    ):
        """持久化场景计划到 scene_execution_units 表。"""
        from datetime import datetime

        def _json_serializer(obj):
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
        logger.info(f"Persisted {len(scenes)} scenes to scene_execution_units")