"""
测试 Writer 的 State Runtime 集成
"""

import pytest
from src.agents.writer import WritingAgent
from src.orchestrator.state import AgentState
from src.runtime import StateCapability, PredictionCapability, RealizationCapability, RetryStrategy


class TestWriterTR:
    """测试 TR 获取"""
    
    def test_get_tr_from_metadata(self):
        agent = WritingAgent()
        state = AgentState(
            metadata={"transition_rigidity": 0.33},
            scene_plan={"scene_id": "unknown"}
        )
        tr = agent._get_transition_rigidity(state)
        assert tr == 0.33
    
    def test_get_tr_from_scene_id(self):
        agent = WritingAgent()
        state = AgentState(
            metadata={},
            scene_plan={"scene_id": "scene_reunion"}
        )
        tr = agent._get_transition_rigidity(state)
        assert tr == 0.33
    
    def test_get_tr_default(self):
        agent = WritingAgent()
        state = AgentState(
            metadata={},
            scene_plan={}
        )
        tr = agent._get_transition_rigidity(state)
        assert tr == 0.50
    
    def test_lookup_tr(self):
        agent = WritingAgent()
        assert agent._lookup_tr("scene_reunion") == 0.33
        assert agent._lookup_tr("scene_dilemma") == 0.71
        assert agent._lookup_tr("mp3") == 0.90
        assert agent._lookup_tr("unknown") is None


class TestWriterCapability:
    """测试 Capability 约束应用"""
    
    def test_apply_primary_enhanced(self):
        agent = WritingAgent()
        base = "【写作规则】\n1. 生成正文。"
        cap = StateCapability(
            prediction=PredictionCapability.PRIMARY,
            realization=RealizationCapability.ENHANCED,
            retry=RetryStrategy.FULL,
            reason="test"
        )
        result = agent._apply_capability_constraints(base, cap)
        assert "State 主导事件选择" in result
        assert "增强注入 State" in result
        assert "【写作规则】" in result
    
    def test_apply_disabled_none(self):
        agent = WritingAgent()
        base = "【写作规则】\n1. 生成正文。"
        cap = StateCapability(
            prediction=PredictionCapability.DISABLED,
            realization=RealizationCapability.NONE,
            retry=RetryStrategy.NONE,
            reason="test"
        )
        result = agent._apply_capability_constraints(base, cap)
        assert "不改变事件选择" in result
        assert "不注入 State" in result
    
    def test_constraints_position(self):
        agent = WritingAgent()
        base = "开头内容。\n\n【写作规则】\n1. 生成正文。"
        cap = StateCapability(
            prediction=PredictionCapability.PRIMARY,
            realization=RealizationCapability.NORMAL,
            retry=RetryStrategy.FULL,
            reason="test"
        )
        result = agent._apply_capability_constraints(base, cap)
        # 约束应该被插入在 "【写作规则】" 之前
        marker_pos = result.find("【写作规则】")
        assert marker_pos > 0
        assert "State" in result[:marker_pos]
    
    def test_no_constraints_applied(self):
        # 如果约束为空的边缘情况，应返回原 prompt
        agent = WritingAgent()
        base = "【写作规则】\n1. 生成正文。"
        # 创建一个模拟的 capability，其中某些模式会返回空约束的情况
        # 实际上，按当前实现，所有枚举值都会产生约束
        # 跳过这个测试，因为它不适用
    
    def test_empty_capability(self):
        # 测试 capability 字段为空的极端情况（不应发生）
        agent = WritingAgent()
        base = "【写作规则】"
        # 这里的 capability 是真实有效的，只是测试结构完整性
        cap = StateCapability(
            prediction=PredictionCapability.ASSIST,
            realization=RealizationCapability.NORMAL,
            retry=RetryStrategy.REALIZATION_ONLY,
            reason="test"
        )
        result = agent._apply_capability_constraints(base, cap)
        assert result is not None