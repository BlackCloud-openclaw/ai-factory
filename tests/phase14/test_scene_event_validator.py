# tests/phase14/test_scene_event_validator.py
"""
Phase 14.0A-1: SceneEventValidator v2 Final 测试
"""

import pytest
from src.writing.scene_event_validator import (
    SceneEventValidator,
    SceneEventRequirement,
    EventValidationStatus,
)


class TestSceneEventValidator:
    """基本功能测试"""

    def test_valid_event(self):
        result = SceneEventValidator.validate_event("发现九宫阵图残片")
        assert result.status == EventValidationStatus.VALID
        assert result.has_action is True
        assert result.has_target is True

    def test_valid_event_generic(self):
        # 不依赖修仙名词
        result = SceneEventValidator.validate_event("发现黑洞形成规律")
        assert result.status == EventValidationStatus.VALID

    def test_valid_event_research(self):
        result = SceneEventValidator.validate_event("科学家确认引力波信号")
        assert result.status == EventValidationStatus.VALID

    def test_valid_event_relationship(self):
        result = SceneEventValidator.validate_event("两国使者达成和平协议")
        assert result.status == EventValidationStatus.VALID

    def test_valid_event_political(self):
        result = SceneEventValidator.validate_event("说服宗门长老改变立场")
        assert result.status == EventValidationStatus.VALID

    def test_valid_event_observation_with_transition(self):
        result = SceneEventValidator.validate_event("确认阵纹变化与灵脉关联")
        assert result.status == EventValidationStatus.VALID

    def test_warning_abstract_target(self):
        result = SceneEventValidator.validate_event("推动剧情发展")
        # 有动作但目标抽象 → WARNING 或 INVALID
        assert result.status != EventValidationStatus.VALID

    def test_warning_implicit_transition(self):
        result = SceneEventValidator.validate_event("观察阵纹变化")
        # 有动作和对象，但转换不明确 → WARNING
        assert result.status == EventValidationStatus.WARNING

    def test_invalid_forbidden_pattern(self):
        result = SceneEventValidator.validate_event("推进主线剧情（场景1）")
        assert result.status == EventValidationStatus.INVALID
        assert any("占位符" in i for i in result.issues)

    def test_invalid_missing_action(self):
        result = SceneEventValidator.validate_event("九宫阵图残片")
        assert result.status == EventValidationStatus.INVALID
        assert any("动作" in i for i in result.issues)

    def test_invalid_missing_target(self):
        result = SceneEventValidator.validate_event("发现")
        assert result.status == EventValidationStatus.INVALID
        assert any("目标" in i for i in result.issues)

    def test_invalid_missing_transition(self):
        result = SceneEventValidator.validate_event("查看石碑")
        assert result.status == EventValidationStatus.INVALID

    def test_scene_validation_valid(self):
        events = [
            "发现九宫阵图残缺部分",
            "获得上古灵丹秘方",
        ]
        result = SceneEventValidator.validate_scene(events)
        assert result.valid is True
        assert result.valid_count == 2
        assert result.contract_quality == "complete"
        assert result.blocking_errors == 0

    def test_scene_validation_with_warning(self):
        events = [
            "观察阵纹变化",  # WARNING
            "获得玉佩",      # VALID
        ]
        result = SceneEventValidator.validate_scene(events)
        # 有 WARNING 但无 INVALID，整体通过
        assert result.valid is True
        assert result.warning_count >= 1
        assert result.contract_quality == "partial"
        assert result.blocking_errors == 0

    def test_scene_validation_invalid(self):
        events = [
            "推进主线剧情",  # INVALID
            "获得玉佩",      # VALID
        ]
        result = SceneEventValidator.validate_scene(events)
        assert result.valid is False
        assert result.invalid_count >= 1
        assert result.contract_quality == "invalid"
        assert result.blocking_errors >= 1

    def test_scene_validation_empty(self):
        events = []
        result = SceneEventValidator.validate_scene(events)
        assert result.valid is False
        assert result.contract_quality == "empty"
        assert result.blocking_errors == 0

    def test_get_invalid_events(self):
        events = ["推进主线剧情", "获得玉佩"]
        result = SceneEventValidator.validate_scene(events)
        invalid = SceneEventValidator.get_invalid_events(result)
        assert "推进主线剧情" in invalid

    def test_has_blocking_issues(self):
        events = ["推进主线剧情", "获得玉佩"]
        result = SceneEventValidator.validate_scene(events)
        assert SceneEventValidator.has_blocking_issues(result) is True

    def test_scene_validation_mixed(self):
        """测试混合状态场景"""
        events = [
            "推进主线剧情",   # INVALID
            "观察阵纹变化",   # WARNING
            "获得玉佩",       # VALID
        ]
        result = SceneEventValidator.validate_scene(events)
        # 有 INVALID，整体失败
        assert result.valid is False
        assert result.contract_quality == "invalid"
        assert result.blocking_errors == 1
        assert result.valid_count == 1
        assert result.warning_count == 1
        assert result.invalid_count == 1


class TestSceneEventRequirement:
    """自定义要求测试"""

    def test_custom_minimum_count(self):
        req = SceneEventRequirement(minimum_count=3)
        events = [
            "发现玉佩",
            "进入禁地",
        ]
        result = SceneEventValidator.validate_scene(events, req)
        assert result.valid is False
        assert "事件数量不足" in result.summary

    def test_custom_forbidden_patterns(self):
        req = SceneEventRequirement(
            forbidden_patterns=[r"测试.*占位符"]
        )
        result = SceneEventValidator.validate_event("测试占位符事件", req)
        assert result.status == EventValidationStatus.INVALID
        assert any("占位符" in i for i in result.issues)


class TestEventStructureDetector:
    """结构检测器测试"""

    def test_detect_action_with_verb(self):
        # 需要有 jieba 环境
        result = SceneEventValidator.validate_event("发现古墓")
        assert result.has_action is True

    def test_detect_action_without_verb(self):
        result = SceneEventValidator.validate_event("古墓")
        assert result.has_action is False

    def test_detect_target_with_noun(self):
        result = SceneEventValidator.validate_event("获得玉佩")
        assert result.has_target is True

    def test_detect_transition_explicit(self):
        result = SceneEventValidator.validate_event("发现古墓")
        assert result.has_transition is True
        assert result.transition_type == "explicit"

    def test_detect_transition_implicit(self):
        result = SceneEventValidator.validate_event("意识到危机")
        assert result.has_transition is True
        assert result.transition_type == "implicit"


class TestInvalidSceneContract:
    """异常类测试"""

    def test_invalid_contract_exception(self):
        from src.writing.contracts.exceptions import InvalidSceneContract

        exc = InvalidSceneContract(
            scene_index=0,
            invalid_events=["推进剧情", "制造悬念"],
            message="测试异常"
        )

        assert exc.scene_index == 0
        assert len(exc.invalid_events) == 2
        assert "测试异常" in str(exc)

        dict_repr = exc.to_dict()
        assert dict_repr["scene_index"] == 0
        assert "推进剧情" in dict_repr["invalid_events"]