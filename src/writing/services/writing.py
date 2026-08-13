import json
import re
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.db import get_db_pool
from src.agents.writer import WritingAgent
from src.orchestrator.state_patch import StatePatch, WorkflowPhase
from src.writing.world_state import WorldState
from src.writing.context_compiler import ContextCompiler
from src.writing.voiceprint import VoiceprintRegistry
from .models import WritingCommand, WritingResult

logger = logging.getLogger(__name__)


class WritingService:
    @staticmethod
    async def execute(cmd: WritingCommand) -> WritingResult:
        """执行场景写作事务"""
        # 1. 获取声纹注册表
        voiceprint_registry = VoiceprintRegistry(cmd.voiceprint_config_path)

        # 2. 获取当前世界状态
        world_state = WorldState.from_dict(cmd.current_state) if cmd.current_state else WorldState()

        # 3. 编译上下文
        compiler = ContextCompiler(max_tokens=2000)
        compiled_context = compiler.compile(
            world_state,
            active_characters=cmd.scene_plan.get("characters", []),
            max_active=10
        )

        # 4. 构建 prompt（复用 WritingAgent 的逻辑，但这里直接构建）
        from src.orchestrator.state import AgentState
        temp_state = AgentState(
            novel_id=cmd.novel_id,
            current_volume=cmd.volume,
            current_chapter=cmd.chapter,
            current_scene_index=cmd.scene_idx,
            scene_plan=cmd.scene_plan,
            current_state=cmd.current_state,
            # ========== 传递 Director 输出 ==========
            narrative_blueprint=cmd.narrative_blueprint,
            knowledge_deltas=cmd.knowledge_deltas,
            character_intent=cmd.character_intent,
            # ====== 新增：传递戏剧结构 ======
            drama_structure=cmd.drama_structure,
            metadata=cmd.metadata or {},  # 添加这一行
            planning_contract=cmd.execution_contract,   # D.4.1-a
        )
        # ========== D.4.1-a 边界日志 ==========
        logger.critical(
            "WRITING_SERVICE_AGENT_STATE_CONTRACT: exists=%s type=%s",
            cmd.execution_contract is not None,
            type(cmd.execution_contract).__name__ if cmd.execution_contract is not None else "None"
        )
        # 注入反馈
        if cmd.writing_feedback:
            temp_state.metadata["writing_feedback"] = cmd.writing_feedback

        # ========== P1 诊断：Service 层 Contract 传递 ==========
        planning_contract = getattr(cmd, "execution_contract", None)
        if planning_contract:
            # 提取 contract_id（兼容 dict 和对象）
            contract_id = (
                planning_contract.get("scene_id", "unknown")
                if isinstance(planning_contract, dict)
                else getattr(planning_contract, "scene_id", "unknown")
            )
            # 提取 state_changes
            if isinstance(planning_contract, dict):
                obs = planning_contract.get("observables", {})
                scs = obs.get("state_changes", [])
            else:
                scs = planning_contract.observables.state_changes if hasattr(planning_contract, 'observables') else []
            
            logger.critical(
                "WRITER_CONTRACT_INPUT: type=%s contract_id=%s state_changes=%s",
                type(planning_contract).__name__,
                contract_id,
                [
                    {"type": sc.get("type") if isinstance(sc, dict) else getattr(sc, "type", None)}
                    for sc in scs
                ]
            )
        else:
            logger.critical("WRITER_CONTRACT_INPUT: planning_contract is None")
        # ===========================================================

        writer = WritingAgent()
        result = await writer.run(temp_state)
        raw_json = result.get("scene_text", "")
        if not raw_json:
            return WritingResult(
                state_patch=StatePatch(error="No output from writer"),
                scene_text="",
                events=[],
                error="No output from writer"
            )

        # 5. 解析 JSON 提取 scene_text 和 events
        clean_text = ""
        events = []
        deviation_detected = result.get("deviation_detected", False)
        missing_goal = result.get("missing_goal_keywords", [])
        missing_conflict = result.get("missing_conflict_keywords", [])

        json_match = re.search(r'\{.*\}', raw_json, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                clean_text = data.get("scene_text", "")
                events = data.get("events", [])
            except:
                clean_text = raw_json
        else:
            clean_text = raw_json
            
        # 7. 构建 StatePatch
        patch = StatePatch(
            scene_text=raw_json,          # 原始输出
            final_answer=clean_text,      # 提取的正文
            deviation_detected=deviation_detected,
            missing_goal_keywords=missing_goal,
            missing_conflict_keywords=missing_conflict,
        )

        return WritingResult(
            state_patch=patch,
            scene_text=clean_text,
            events=events,
            deviation_detected=deviation_detected,
            missing_goal_keywords=missing_goal,
            missing_conflict_keywords=missing_conflict,
        )