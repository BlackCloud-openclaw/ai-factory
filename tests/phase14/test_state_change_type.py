# tests/phase14/test_state_change_type.py
"""
Phase 14.0A-2 / 14.0B-1: StateChangeType 枚举验证
"""

import pytest
from src.writing.state_change_types import StateChangeType


class TestStateChangeType:
    def test_enum_count(self):
        """ADR-049.2-C2: StateChange 类型 ≤6"""
        assert StateChangeType.count() == 6

    def test_enum_values(self):
        """验证枚举值完整性"""
        expected = {
            "knowledge_gain",
            "inventory_acquire",
            "location_change",
            "realm_change",
            "relationship_change",
            "plot_flag",
        }
        assert set(StateChangeType.values()) == expected

    def test_enum_no_extra(self):
        """确保没有额外值漂移"""
        assert len(StateChangeType) == 6