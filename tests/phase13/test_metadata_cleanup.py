# tests/phase13/test_metadata_cleanup.py
"""
Phase 13.2.3D: Metadata Cleanup 验证测试
确保核心业务状态不在 metadata 中传递。
"""

import pytest
from src.orchestrator.state import AgentState


class TestMetadataCleanup:
    def test_core_state_not_in_metadata(self):
        """确保核心状态字段不在 metadata 中（仅作为调试辅助）。"""
        state = AgentState()
        
        # 这些键不应出现在 metadata 中
        assert "planner_outputs" not in state.metadata
        assert "narrative_intent" not in state.metadata
        assert "scene_plan_list" not in state.metadata

    def test_core_state_none_vs_empty(self):
        """验证 None 和 [] 的语义区别。"""
        state = AgentState()
        
        # 初始状态为 None 或默认值
        # planner_outputs 默认是 []
        assert state.planner_outputs == []
        
        # 显式设置为 None
        state.planner_outputs = None
        assert state.planner_outputs is None
        
        # 显式设置为空列表
        state.planner_outputs = []
        assert state.planner_outputs is not None
        assert len(state.planner_outputs) == 0

    def test_scene_plan_list_none_vs_empty(self):
        """验证 scene_plan_list 的 None/[] 语义。"""
        state = AgentState()
        
        # 默认是 []
        assert state.scene_plan_list == []
        
        state.scene_plan_list = None
        assert state.scene_plan_list is None
        
        state.scene_plan_list = []
        assert state.scene_plan_list is not None
        assert len(state.scene_plan_list) == 0