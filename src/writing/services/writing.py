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
        # 为了避免重复代码，我们可以临时构造一个 AgentState 并调用 WritingAgent.run
        # 但为了服务独立性，我们将 prompt 构建逻辑内联或调用独立函数。
        # 为简单起见，仍然复用 WritingAgent，但不通过节点，直接实例化并传入所需字段。
        # 注意：WritingAgent.run 期望接收一个完整的 AgentState，我们需要构造一个最小状态。

        from src.orchestrator.state import AgentState
        temp_state = AgentState(
            novel_id=cmd.novel_id,
            current_volume=cmd.volume,
            current_chapter=cmd.chapter,
            current_scene_index=cmd.scene_idx,
            scene_plan=cmd.scene_plan,
            current_state=cmd.current_state,
            metadata={},
        )
        # 注入反馈
        if cmd.writing_feedback:
            temp_state.metadata["writing_feedback"] = cmd.writing_feedback

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
            
        # ===== 添加重要性修复 =====
        if events:
            for evt in events:
                if evt.get("type") == "discovery" and "importance" in evt:
                    imp = evt["importance"]
                    if isinstance(imp, int):
                        if imp >= 5:
                            evt["importance"] = "critical"
                        elif imp >= 3:
                            evt["importance"] = "high"
                        elif imp >= 1:
                            evt["importance"] = "normal"
                        else:
                            evt["importance"] = "low"
                    elif isinstance(imp, float):
                        evt["importance"] = "critical" if imp >= 5 else "high" if imp >= 3 else "normal" if imp >= 1 else "low"
                    elif isinstance(imp, bool):
                        evt["importance"] = "critical" if imp else "low"            

        # 6. 更新 scene_execution_units 状态为 running（可选，但可以在节点中做，也可以在这里做）
        # 为了保持服务原子性，这里只返回 patch，不在 service 中直接修改数据库。
        # 节点中会单独调用 _update_scene_unit_status，但我们可以将状态更新也纳入服务吗？
        # 按照设计，service 应该是完整的事务边界。但 writing 本身不改变持久化状态（除了更新 running 状态），
        # 这个更新更适合放在节点中，因为不是事务核心。我们保持原方案：节点中更新 running 状态。

        # 7. 构建 StatePatch
        patch = StatePatch(
            scene_text=raw_json,          # 原始输出
            final_answer=clean_text,      # 提取的正文
            deviation_detected=deviation_detected,
            missing_goal_keywords=missing_goal,
            missing_conflict_keywords=missing_conflict,
            # 注意：写作后不改变 phase，保持在 WRITING 等待验证
        )

        return WritingResult(
            state_patch=patch,
            scene_text=clean_text,
            events=events,
            deviation_detected=deviation_detected,
            missing_goal_keywords=missing_goal,
            missing_conflict_keywords=missing_conflict,
        )