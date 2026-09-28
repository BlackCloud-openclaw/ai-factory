# tests/writing/test_structural_lock.py
"""
Phase 15.8 Commit 2 (calibrated): Structural Lock 单测。

精简为 2 条规则后，4 个有效用例。
"""
import pytest
from src.writing.structural_lock import StructuralLock


def _contract():
    return {
        "scene_context": {"characters": ["林逸", "前代丹尊残魂"]},
        "execution": {"units": [{"description": "遇见被困前代丹尊", "id": "U1", "label": "action"}]},
        "observables": {"state_changes": [{"type": "plot_flag", "name": "禁制核心激活"}]},
    }


ORIG = (
    "林逸在禁制核心遇见了前代丹尊残魂。"
    "残魂低声说道：禁制核心激活了，你必须取得残缺解禁咒文。"
    "林逸点头，记下了这句话。"
)

REWRITTEN_OK = (
    "禁制核心的幽蓝光芒中，林逸的身影微微一顿。"
    "眼前浮现出前代丹尊残魂的虚影，低语似从远处传来。"
    "「禁制核心激活了。」残魂的声音很轻，"
    "「去找残缺解禁咒文。」林逸沉默着记下。"
)


def test_1_identical_passes():
    """rewritten=original → safe"""
    r = StructuralLock().check(ORIG, ORIG, _contract())
    assert r.structural_safe is True
    assert len(r.checks) == 2


def test_2_rewritten_ok_passes():
    """改写句式但保留所有角色 → safe"""
    r = StructuralLock().check(ORIG, REWRITTEN_OK, _contract())
    assert r.structural_safe is True


def test_3_missing_character_fails():
    """删除一个角色名 → characters_preserved fail"""
    bad = REWRITTEN_OK.replace("前代丹尊残魂", "那个虚影")
    r = StructuralLock().check(ORIG, bad, _contract())
    assert r.structural_safe is False
    assert any(c.name == "characters_preserved" and not c.passed for c in r.checks)


def test_4_too_short_fails():
    """rewritten 只有原文 30% 长度 → length_floor fail"""
    bad = "林逸" * 5
    r = StructuralLock().check(ORIG, bad, _contract())
    assert r.structural_safe is False
    assert any(c.name == "length_floor" and not c.passed for c in r.checks)


def test_5_degenerate_contract_passes():
    """contract=None → characters 视为空 → safe"""
    r = StructuralLock().check(ORIG, REWRITTEN_OK, None)
    assert r.structural_safe is True


@pytest.mark.asyncio
async def test_6_async_smoke_degenerate():
    """contract=None → 语义检查直接跳过，不调 embedding"""
    r = await StructuralLock().check_async(ORIG, REWRITTEN_OK, None)
    assert r.structural_safe is True
    # 2 sync + 1 semantic(skipped)
    assert len(r.checks) == 3
    names = {c.name for c in r.checks}
    assert "key_events_semantically_preserved" in names


@pytest.mark.asyncio
async def test_7_async_early_exit_on_sync_fail():
    """sync 规则失败时，async 早退（不调 embedding）"""
    bad = "林逸" * 5   # 长度不够
    r = await StructuralLock().check_async(ORIG, bad, _contract())
    assert r.structural_safe is False
    # 早退，只有 2 条 sync 规则
    assert len(r.checks) == 2