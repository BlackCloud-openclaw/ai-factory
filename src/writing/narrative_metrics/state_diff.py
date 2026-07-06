"""
世界状态差异计算 - 对比前后 WorldState，计算关系/目标/角色变化
"""

from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass


@dataclass
class StateDiffResult:
    """状态差异结果"""
    relationship_delta: float          # 关系变化总量
    relationship_change_count: int     # 有多少对关系发生了变化
    goal_state_changed: bool          # 目标是否改变
    cognitive_model_changed: bool     # 认知模型是否改变
    behavior_changed: bool            # 行为是否改变
    identity_changed: bool            # 身份是否改变


class StateDiffAnalyzer:
    """分析 WorldState 前后差异"""

    @staticmethod
    def analyze(
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        protagonist: str = "林逸",
    ) -> StateDiffResult:
        """
        对比前后状态，计算差异指标
        state 格式参考 WorldState.to_dict()
        """

        # ----- 1. 关系变化 -----
        rel_before = state_before.get("relationships", {})
        rel_after = state_after.get("relationships", {})

        all_keys = set(rel_before.keys()) | set(rel_after.keys())
        total_delta = 0.0
        change_count = 0

        for key in all_keys:
            val_before = rel_before.get(key, 0)
            val_after = rel_after.get(key, 0)
            diff = abs(val_after - val_before)
            if diff > 0.1:
                change_count += 1
                total_delta += diff

        # 归一化 delta 到 0-1（假设最大变化 100）
        normalized_delta = min(1.0, total_delta / 100.0)

        # ----- 2. 目标变化检测 -----
        goal_changed = False

        # 检测 2.1：全局标记变化（如 plot_flag_set）
        flags_before = set(state_before.get("global_flags", {}).keys())
        flags_after = set(state_after.get("global_flags", {}).keys())
        if flags_before != flags_after:
            goal_changed = True

        # 检测 2.2：主角目标标记
        for flag in ["目标已改", "任务转向", "真相大白"]:
            if state_after.get("global_flags", {}).get(flag):
                goal_changed = True

        # ----- 3. 角色认知/行为变化 -----
        cognitive_changed = False
        behavior_changed = False
        identity_changed = False

        # 检测角色认知字段变化
        char_before = state_before.get("characters", {}).get(protagonist, {})
        char_after = state_after.get("characters", {}).get(protagonist, {})

        # 3.1 beliefs 变化（认知模型）
        beliefs_before = set(char_before.get("beliefs", []))
        beliefs_after = set(char_after.get("beliefs", []))
        if beliefs_before != beliefs_after:
            cognitive_changed = True

        # 3.2 self_image 变化（身份认同）
        if char_before.get("self_image") != char_after.get("self_image"):
            identity_changed = True

        # 3.3 moral_boundaries 变化（行为准则）
        if char_before.get("moral_boundaries") != char_after.get("moral_boundaries"):
            behavior_changed = True

        return StateDiffResult(
            relationship_delta=normalized_delta,
            relationship_change_count=change_count,
            goal_state_changed=goal_changed,
            cognitive_model_changed=cognitive_changed,
            behavior_changed=behavior_changed,
            identity_changed=identity_changed,
        )