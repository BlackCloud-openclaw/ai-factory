# tests/integration/remote/gc/test_models.py
"""
B4.10: MarkerReconciliationResult 单元测试（集成环境）
"""

import pytest
from src.writing.snapshot.runtime.remote.gc import MarkerReconciliationResult


class TestMarkerReconciliationResult:

    def test_valid_creation(self):
        """正常创建 MarkerReconciliationResult 实例。"""
        result = MarkerReconciliationResult(
            scanned_markers=10,
            stale_found=5,
            stale_cleared=3,
            errors=2,
            protected_found=1,
            protected_cleared=1,
        )
        assert result.scanned_markers == 10
        assert result.stale_found == 5
        assert result.stale_cleared == 3
        assert result.errors == 2
        assert result.protected_found == 1
        assert result.protected_cleared == 1

    def test_post_init_validates_cleared_not_exceed_found(self):
        """stale_cleared 不能超过 stale_found。"""
        with pytest.raises(ValueError, match="stale_cleared .* cannot exceed stale_found"):
            MarkerReconciliationResult(
                stale_found=5,
                stale_cleared=6,
            )

    def test_post_init_validates_protected_cleared_not_exceed_found(self):
        """protected_cleared 不能超过 protected_found。"""
        with pytest.raises(ValueError, match="protected_cleared .* cannot exceed protected_found"):
            MarkerReconciliationResult(
                protected_found=2,
                protected_cleared=3,
            )

    def test_post_init_validates_non_negative_fields(self):
        """所有字段必须是非负数。"""
        with pytest.raises(ValueError, match="scanned_markers must be non-negative"):
            MarkerReconciliationResult(scanned_markers=-1)

        with pytest.raises(ValueError, match="stale_found must be non-negative"):
            MarkerReconciliationResult(stale_found=-1)

        with pytest.raises(ValueError, match="stale_cleared must be non-negative"):
            MarkerReconciliationResult(stale_cleared=-1)

        with pytest.raises(ValueError, match="errors must be non-negative"):
            MarkerReconciliationResult(errors=-1)

        with pytest.raises(ValueError, match="protected_found must be non-negative"):
            MarkerReconciliationResult(protected_found=-1)

        with pytest.raises(ValueError, match="protected_cleared must be non-negative"):
            MarkerReconciliationResult(protected_cleared=-1)

    def test_properties(self):
        """验证属性计算正确。"""
        result = MarkerReconciliationResult(
            scanned_markers=20,
            stale_found=10,
            stale_cleared=7,
            errors=2,
            protected_found=1,
            protected_cleared=0,
        )
        assert result.issues_found == 11  # stale_found + protected_found
        assert result.issues_fixed == 7   # stale_cleared + protected_cleared
        assert result.issues_remaining == 6  # (10-7) + (1-0) + 2

    def test_str(self):
        """字符串表示正确。"""
        result = MarkerReconciliationResult(
            scanned_markers=100,
            stale_found=50,
            stale_cleared=30,
            errors=5,
            protected_found=2,
            protected_cleared=2,
        )
        expected = (
            "MarkerReconciliationResult("
            "scanned=100, "
            "stale_found=50, "
            "stale_cleared=30, "
            "protected_found=2, "
            "protected_cleared=2, "
            "errors=5)"
        )
        assert str(result) == expected

    def test_defaults(self):
        """默认值验证。"""
        result = MarkerReconciliationResult()
        assert result.scanned_markers == 0
        assert result.stale_found == 0
        assert result.stale_cleared == 0
        assert result.errors == 0
        assert result.protected_found == 0
        assert result.protected_cleared == 0
        assert result.issues_found == 0
        assert result.issues_fixed == 0
        assert result.issues_remaining == 0
        assert str(result) == (
            "MarkerReconciliationResult("
            "scanned=0, stale_found=0, stale_cleared=0, "
            "protected_found=0, protected_cleared=0, errors=0)"
        )

    def test_issues_remaining_with_only_stale(self):
        """仅有 stale 标记时的剩余问题数。"""
        result = MarkerReconciliationResult(
            stale_found=5,
            stale_cleared=3,
        )
        assert result.issues_remaining == 2

    def test_issues_remaining_with_only_protected(self):
        """仅有 protected 标记时的剩余问题数。"""
        result = MarkerReconciliationResult(
            protected_found=4,
            protected_cleared=1,
        )
        assert result.issues_remaining == 3

    def test_issues_remaining_with_both_and_errors(self):
        """混合场景。"""
        result = MarkerReconciliationResult(
            stale_found=8,
            stale_cleared=5,
            protected_found=3,
            protected_cleared=2,
            errors=1,
        )
        assert result.issues_remaining == (8-5) + (3-2) + 1 == 5