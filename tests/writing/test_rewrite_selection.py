# tests/writing/test_rewrite_selection.py
"""
Phase 15.8 Commit 1: Rewrite Selection Contract 单测。

全部使用 mock，不调 LLM，不写数据库。
覆盖 6 个决策分支，验证 Commit 1 硬不变量。
"""
import pytest
from unittest.mock import AsyncMock

from src.writing.controlled_writer import (
    ControlledWriter,
    RewriteSelectionReason,
)
from src.writing.planning_contract import (
    PlanningContract, Intent, Execution, ExecutionUnit, Observables,
    ContractMetadata,
)


def _make_contract() -> PlanningContract:
    return PlanningContract(
        scene_id="test_scene",
        intent=Intent(goal="测试目标", conflict="测试冲突", expected_outcome="测试结果"),
        execution=Execution(units=[ExecutionUnit(id="U1", label="action", description="测试单元")]),
        observables=Observables(),
        metadata=ContractMetadata(chapter=1, scene_index=0),
    )


def _long_text() -> str:
    return "这是一段足够长的测试文本。" * 20  # > 50 chars


@pytest.mark.asyncio
async def test_branch_1_no_rewriter_injected():
    """rewriter=None → no_rewriter_injected"""
    cw = ControlledWriter(rewriter=None)
    selection, rewritten = await cw._select_rewrite(
        original_text=_long_text(),
        execution_contract=_make_contract(),
    )
    assert selection.selected_source == "original"
    assert selection.selection_reason == RewriteSelectionReason.NO_REWRITER_INJECTED.value
    assert selection.structural_safe is False
    assert rewritten is None


@pytest.mark.asyncio
async def test_branch_2_text_too_short():
    """text < 50 chars → text_too_short"""
    cw = ControlledWriter(rewriter=AsyncMock())
    selection, rewritten = await cw._select_rewrite(
        original_text="短",
        execution_contract=_make_contract(),
    )
    assert selection.selection_reason == RewriteSelectionReason.TEXT_TOO_SHORT.value
    assert selection.structural_safe is False
    assert rewritten is None


@pytest.mark.asyncio
async def test_branch_3_missing_execution_contract():
    """execution_contract=None → missing_execution_contract"""
    cw = ControlledWriter(rewriter=AsyncMock())
    selection, rewritten = await cw._select_rewrite(
        original_text=_long_text(),
        execution_contract=None,
    )
    assert selection.selection_reason == RewriteSelectionReason.MISSING_EXECUTION_CONTRACT.value
    assert selection.structural_safe is False
    assert rewritten is None


@pytest.mark.asyncio
async def test_branch_4_rewrite_raises():
    """rewriter raises → rewrite_unavailable"""
    mock_rewriter = AsyncMock()
    mock_rewriter.rewrite = AsyncMock(side_effect=RuntimeError("boom"))
    cw = ControlledWriter(rewriter=mock_rewriter)
    selection, rewritten = await cw._select_rewrite(
        original_text=_long_text(),
        execution_contract=_make_contract(),
    )
    assert selection.selection_reason == RewriteSelectionReason.REWRITE_UNAVAILABLE.value
    assert selection.structural_safe is False
    assert selection.rewrite_attempted is True
    assert "boom" in (selection.rewrite_failure_reason or "")
    assert rewritten is None


@pytest.mark.asyncio
async def test_branch_5_rewrite_returns_blank():
    """rewriter returns blank → rewrite_unavailable"""
    mock_rewriter = AsyncMock()
    mock_rewriter.rewrite = AsyncMock(return_value="   ")
    cw = ControlledWriter(rewriter=mock_rewriter)
    selection, rewritten = await cw._select_rewrite(
        original_text=_long_text(),
        execution_contract=_make_contract(),
    )
    assert selection.selection_reason == RewriteSelectionReason.REWRITE_UNAVAILABLE.value
    assert selection.structural_safe is False
    assert selection.rewrite_attempted is True
    assert rewritten is None


@pytest.mark.asyncio
async def test_branch_6_flip_to_rewritten(monkeypatch):
    """Phase 15.8 Commit 3B: structural_safe=True → selected_source='rewritten'"""
    mock_rewriter = AsyncMock()
    mock_rewriter.rewrite = AsyncMock(return_value="改写后的文本。" * 30)

    # mock StructuralLock.check_async 恒返回 safe（不依赖 embedding 服务）
    async def _fake_check_async(self, original_text, rewritten_text, contract):
        from src.writing.structural_lock import LockResult, LockCheck
        return LockResult(
            structural_safe=True,
            checks=[LockCheck("mocked", True, "test")],
            failure_summary="",
        )

    from src.writing.structural_lock import StructuralLock
    monkeypatch.setattr(StructuralLock, "check_async", _fake_check_async)

    cw = ControlledWriter(rewriter=mock_rewriter)
    selection, rewritten = await cw._select_rewrite(
        original_text=_long_text(),
        execution_contract=_make_contract(),
    )
    assert selection.selected_source == "rewritten"
    assert selection.selection_reason == RewriteSelectionReason.SELECTED.value
    assert selection.structural_safe is True
    assert selection.rewrite_available is True
    assert selection.rewrite_attempted is True
    assert rewritten is not None


@pytest.mark.asyncio
async def test_branch_7_flip_to_original_when_unsafe(monkeypatch):
    """Phase 15.8 Commit 3B: structural_safe=False → selected_source='original'"""
    mock_rewriter = AsyncMock()
    mock_rewriter.rewrite = AsyncMock(return_value="改写后的文本。" * 30)

    async def _fake_check_async(self, original_text, rewritten_text, contract):
        from src.writing.structural_lock import LockResult, LockCheck
        return LockResult(
            structural_safe=False,
            checks=[LockCheck("mocked", False, "unsafe")],
            failure_summary="mocked unsafe",
        )

    from src.writing.structural_lock import StructuralLock
    monkeypatch.setattr(StructuralLock, "check_async", _fake_check_async)

    cw = ControlledWriter(rewriter=mock_rewriter)
    selection, rewritten = await cw._select_rewrite(
        original_text=_long_text(),
        execution_contract=_make_contract(),
    )
    assert selection.selected_source == "original"
    assert selection.selection_reason == RewriteSelectionReason.STRUCTURAL_UNSAFE.value
    assert selection.structural_safe is False
    assert selection.rewrite_available is True
    assert rewritten is not None