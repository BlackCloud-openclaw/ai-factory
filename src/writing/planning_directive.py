# src/writing/planning_directive.py
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class PlanningRepresentation(str, Enum):
    SUMMARY = "summary"
    BEAT = "beat"
    ACTION = "action"
    INTENT = "intent"
    CONSTRAINT = "constraint"

class PlanningDirective(BaseModel):
    """统一的规划指令模型 - 用于实验"""
    representation: PlanningRepresentation
    content: Dict[str, Any]
    expected_outcome: Optional[Dict[str, Any]] = None  # 预期世界状态变化（用于深层评估）

    @classmethod
    def from_scene_plan(cls, scene_plan: Dict, rep: PlanningRepresentation, density: str = "sparse"):
        """将现有场景计划转换为不同表示（实验核心）"""
        goal = scene_plan.get("goal", "")
        conflict = scene_plan.get("conflict", "")
        outcome = scene_plan.get("outcome", "")
        must_events = scene_plan.get("must_events", [])
        forbidden = scene_plan.get("forbidden_events", [])
        state_delta = scene_plan.get("state_delta", {})

        if rep == PlanningRepresentation.SUMMARY:
            if density == "sparse":
                summary = f"{goal}。面临{conflict}。"
            else:  # dense
                summary = f"场景目标：{goal}。核心冲突：{conflict}。预期结果：{outcome}。"
            return cls(representation=rep, content={"summary": summary}, expected_outcome=state_delta)

        elif rep == PlanningRepresentation.BEAT:
            beats = must_events if must_events else [f"推进{goal}"]
            if density == "dense":
                beats = [f"{i}. {b}（关键节拍）" for i, b in enumerate(beats, 1)]
            return cls(representation=rep, content={"beats": beats}, expected_outcome=state_delta)

        elif rep == PlanningRepresentation.ACTION:
            actions = [f"执行动作：{e}" for e in must_events] if must_events else [f"采取行动达成{goal}"]
            if density == "dense":
                actions = [f"{a}，并产生直接影响" for a in actions]
            return cls(representation=rep, content={"actions": actions}, expected_outcome=state_delta)

        elif rep == PlanningRepresentation.INTENT:
            # 意图式：强调"目的"和"成功条件"
            intent_content = {
                "purpose": goal,
                "success_condition": outcome or f"{goal}完成",
                "constraints": forbidden
            }
            if density == "dense":
                intent_content["sub_goals"] = must_events
                intent_content["failure_condition"] = f"未能{goal}"
            return cls(representation=rep, content=intent_content, expected_outcome=state_delta)

        elif rep == PlanningRepresentation.CONSTRAINT:
            # 约束式：只告诉 Writer 不能做什么和必须做什么
            constraint_content = {
                "must_happen": must_events if must_events else [f"完成{goal}"],
                "must_not_happen": forbidden
            }
            if density == "sparse":
                constraint_content["must_happen"] = constraint_content["must_happen"][:2]  # 只给前2个
            return cls(representation=rep, content=constraint_content, expected_outcome=state_delta)

        # fallback
        return cls(representation=PlanningRepresentation.SUMMARY, content={"summary": goal}, expected_outcome=state_delta)