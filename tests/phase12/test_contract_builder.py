"""
测试 ContractBuilder
"""

import pytest
from experiments.phase12.corpus.models import CorpusSample, FailureMode, Difficulty
from experiments.phase12.corpus.contract_builder import ContractBuilder


def test_builder_creates_contract():
    """验证 Builder 生成有效的 PlanningContract"""
    sample = CorpusSample(
        id="test_sample",
        version="1.0",
        category=FailureMode.SCENE_TRANSITION,
        failure_modes=[FailureMode.SCENE_TRANSITION],
        difficulty=Difficulty.MEDIUM,
        language="zh-CN",
        scene_before="林逸推开石门。",
        scene_after=None,
        draft_before=None,
        draft_after=None,
        expected={},
        artifacts={},
        source="test",
        license="internal",
        tags=(),
    )

    builder = ContractBuilder()
    contract = builder.build(sample)

    assert contract.scene_id == "contract_test_sample"
    assert contract.intent.goal == "完成自然的场景推进"
    assert contract.intent.conflict == "场景转换缺少因果连接"
    assert len(contract.execution.units) == 1


def test_contract_deterministic():
    """验证 Contract 生成是确定性的"""
    sample = CorpusSample(
        id="test_sample",
        version="1.0",
        category="runtime_state",
        failure_modes=["runtime_state"],
        difficulty=Difficulty.MEDIUM,
        language="zh-CN",
        scene_before="状态异常。",
        scene_after=None,
        draft_before=None,
        draft_after=None,
        expected={},
        artifacts={},
        source="test",
        license="internal",
        tags=(),
    )

    builder = ContractBuilder()
    c1 = builder.build(sample)
    c2 = builder.build(sample)

    assert c1.scene_id == c2.scene_id
    assert c1.intent.goal == c2.intent.goal
    assert c1.intent.conflict == c2.intent.conflict


def test_contract_no_evaluation_semantic():
    """
    验证 Contract 不包含评估语义。
    这是防止 Benchmark 概念泄漏到 Writer 的关键测试。
    """
    sample = CorpusSample(
        id="test_sample",
        version="1.0",
        category="dialogue_quality",
        failure_modes=["dialogue_quality"],
        difficulty=Difficulty.MEDIUM,
        language="zh-CN",
        scene_before="对话开始。",
        scene_after=None,
        draft_before=None,
        draft_after=None,
        expected={},
        artifacts={},
        source="test",
        license="internal",
        tags=(),
    )

    contract = ContractBuilder().build(sample)

    description = contract.execution.units[0].description
    assert "评估" not in description
    assert "benchmark" not in description
    assert "测试" not in description
    assert "检查" not in description