# src/writing/scene_event_validator.py
"""
Phase 14.0A-1: SceneEventValidator v2 (Final)

验证事件是否具备可验证的叙事结构，不依赖题材领域知识。
只检查：动作（action）、目标（target）、状态转换（transition）

ADR-049.3 约束：
- C8: 不判断领域合法性
- C9: action != transition
- C10: Validator 不执行 repair
- C11: fallback 必须保守

Phase 14.0C-2 扩展：
- 扩展 ACTION_KEYWORDS 和 TARGET_KEYWORDS，支持“参悟”、“纹路”、“传承”等修仙常用词
"""

import re
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum

# 尝试导入 jieba，如果不可用则使用简单回退
try:
    import jieba.posseg as pseg
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False


class EventValidationStatus(str, Enum):
    VALID = "valid"          # 结构完整，可直接使用
    WARNING = "warning"      # 结构部分完整，可接受但需注意
    INVALID = "invalid"      # 结构缺失，必须修复


@dataclass
class EventValidationResult:
    """单个事件的验证结果"""
    event: str
    status: EventValidationStatus
    has_action: bool
    has_target: bool
    target_type: Literal["concrete", "abstract", "unknown"]
    has_transition: bool
    transition_type: Literal["explicit", "implicit", "none"]
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class SceneValidationResult:
    """整个场景的验证结果"""
    valid: bool
    events: List[EventValidationResult]
    valid_count: int
    warning_count: int
    invalid_count: int
    contract_quality: Literal["complete", "partial", "invalid", "empty"]
    blocking_errors: int
    summary: str


@dataclass
class SceneEventRequirement:
    """验证要求（可配置）"""
    minimum_count: int = 2
    require_action: bool = True
    require_target: bool = True
    require_transition: bool = True
    allow_warning: bool = True

    forbidden_patterns: List[str] = field(default_factory=lambda: [
        r"推进.*剧情",
        r"推进.*主线",
        r"推动.*故事",
        r"制造.*悬念",
        r"加强.*冲突",
        r"铺垫.*伏笔",
        r"场景\d+",
        r"剧情.*推进",
    ])


class EventStructureDetector:
    """
    事件结构检测器 - 不依赖领域词表，只检测结构模式
    """

    # Phase 14.0A-1: 扩展通用转换动词（不限于修仙题材）
    TRANSITION_VERBS = [
        # 原始修仙/动作类
        "发现", "获得", "捡到", "夺取", "缴获",
        "进入", "抵达", "离开", "返回",
        "突破", "晋升", "突破至",
        "击杀", "击败", "击退", "战胜", "杀死",
        "结盟", "交恶", "决裂", "和解", "背叛",
        "确认", "揭示", "揭露", "验证", "证明",
        "触发", "启动", "激活", "破解", "解除",
        "找到", "寻得", "遇见", "遭遇",
        "阻止", "破坏", "摧毁",
        "说服", "改变", "转换",
        # 通用转换
        "达成", "建立", "签署", "同意", "承诺",
        "开始", "停止", "结束", "完成",
        "成为", "变为", "转为",
        "增加", "减少", "提升", "下降",
        "接受", "拒绝", "承认", "废除",
        # Phase 14.0C-2 扩展（修仙通用）
        "参悟", "领悟", "感悟", "理解", "解析", "破译", "刻录", "铭刻",
        "激活", "唤醒", "释放", "汇聚", "流转", "凝练", "淬炼", "融合",
        "召唤", "驱逐", "镇压", "净化", "封印", "解封", "修复", "重塑",
        "觉醒", "顿悟", "明悟", "勘破", "贯通", "驾驭", "掌控", "蜕变",
        "演化", "升华", "质变", "跃迁",
        # 🆕 本次新增（炼器专用）
        "炼器", "锻造", "铸造", "淬炼", "锻打", "熔炼",
        "反噬", "共鸣", "震荡", "爆发", "湮灭", "吞噬", "剥离", "融合",
        "铭纹", "刻阵", "激活", "立下", "签订",
    ]

    # 抽象过渡词（状态变化不明确）
    ABSTRACT_TRANSITION = [
        "感到", "觉得", "感觉", "认为", "想",
        "意识到", "认识到", "体会到",
        "怀疑", "猜测", "推测",
        "觉得", "认为",
    ]

    # 抽象目标词（无法验证状态变化）
    ABSTRACT_TARGETS = [
        "剧情", "情节", "故事", "发展",
        "冲突", "氛围", "情绪", "感受",
        "关系", "局势", "局面", "形势",
    ]

    # 回退关键词（用于 jieba 不可用时的备用检测）
    ACTION_KEYWORDS = [
        # 原有动作关键词
        "获得", "捡到", "夺取", "发现", "进入", "抵达", "离开", "返回",
        "突破", "晋升", "击杀", "击败", "结盟", "交恶", "决裂", "和解",
        "揭示", "揭露", "确认", "验证", "触发", "启动", "破解",
        "找到", "寻得", "遇见", "遭遇", "阻止", "破坏", "摧毁",
        "说服", "改变", "转换", "达成", "建立", "签署", "同意", "承诺",
        "开始", "停止", "结束", "完成", "成为", "变为", "转为",
        "增加", "减少", "提升", "下降", "接受", "拒绝", "承认", "废除",
        # Phase 14.0C-2 扩展
        "参悟", "领悟", "感悟", "理解", "解析", "破译", "刻录", "铭刻",
        "激活", "唤醒", "释放", "汇聚", "流转", "凝练", "淬炼", "融合",
        "召唤", "驱逐", "镇压", "净化", "封印", "解封", "修复", "重塑",
        "觉醒", "顿悟", "明悟", "勘破", "贯通", "驾驭", "掌控", "蜕变",
        "演化", "升华", "质变", "跃迁",
        # 🆕 本次新增（炼器专用）
        "炼器", "锻造", "铸造", "淬炼", "锻打", "熔炼",
        "反噬", "共鸣", "震荡", "立下", "签订",
    ]

    TARGET_KEYWORDS = [
        # 原有目标词
        "玉佩", "秘境", "丹药", "功法", "灵兽", "法器", "阵法", "灵石",
        "玉简", "丹炉", "药草", "妖兽", "禁制", "符文", "传承", "密室",
        "石室", "灵脉", "仙府", "洞府", "遗址", "剑痕", "血纹", "铭纹",
        # Phase 14.0C-2 已扩展
        "纹路", "石碑", "古籍", "残卷", "卷轴", "印记", "法则", "道韵",
        "灵泉", "丹方", "阵图", "法宝", "灵宝", "仙术", "神通", "剑意",
        "领域", "法相", "元神", "元婴", "金丹", "道基", "神识", "灵识",
        "血珀", "玉符", "阵盘", "阵旗", "灵木", "仙草", "奇花", "异果",
        "混沌", "鸿蒙", "天道", "因果", "轮回", "时空",
        # 🆕 本次新增（针对炼器/灵傀）
        "炼器之道", "炼器术", "锻造术", "铸造术", "淬炼术",
        "灵傀", "器灵", "器魂", "器胚", "剑胚", "法宝雏形",
        "熔炉", "淬炼池", "锻台", "铁砧",
        "灵纹", "器纹", "阵纹", "契约", "誓约", "条约",
    ]

    @classmethod
    def detect_action(cls, text: str) -> bool:
        if not text:
            return False

        # 🔥 优先检查是否包含动作关键词
        for kw in cls.ACTION_KEYWORDS:
            if kw in text:
                return True

        # 如果 jieba 可用，使用词性标注作为补充
        if HAS_JIEBA:
            words = pseg.cut(text)
            return any(word.flag.startswith('v') for word in words)

        return False

    @classmethod
    def detect_target(cls, text: str) -> tuple[bool, Literal["concrete", "abstract", "unknown"]]:
        if not text:
            return False, "unknown"

        # 🔥 优先检查是否包含具体目标关键词（无论 jieba 是否可用）
        for kw in cls.TARGET_KEYWORDS:
            if kw in text:
                return True, "concrete"

        # 检查抽象目标词（如“剧情”、“冲突”）
        for ab in cls.ABSTRACT_TARGETS:
            if ab in text:
                return True, "abstract"

        # 如果 jieba 可用，使用词性标注作为补充
        if HAS_JIEBA:
            words = pseg.cut(text)
            for word in words:
                if word.flag.startswith('n'):
                    # 名词，且不在抽象目标中（已检查过）
                    return True, "concrete"
            return False, "unknown"

        return False, "unknown"

    @classmethod
    def detect_transition(cls, text: str) -> tuple[bool, Literal["explicit", "implicit", "none"]]:
        if not text:
            return False, "none"

        # 1. 显式转换动词
        for verb in cls.TRANSITION_VERBS:
            if verb in text:
                return True, "explicit"

        # 2. 抽象转换词（警告级别）
        for verb in cls.ABSTRACT_TRANSITION:
            if verb in text:
                return True, "implicit"

        # 3. 结构模式："从...到..." 或 "变为..."
        if "从" in text and "到" in text:
            return True, "explicit"
        if "变为" in text or "成为" in text:
            return True, "explicit"

        # 4. Phase 14.0A-1: 识别“变化”等词作为隐性转换
        if "变化" in text or "改变" in text or "变动" in text:
            return True, "implicit"

        return False, "none"


class SceneEventValidator:
    """场景事件验证器 - v2 Final"""

    @classmethod
    def validate_event(
        cls,
        event: str,
        requirement: Optional[SceneEventRequirement] = None,
    ) -> EventValidationResult:
        if requirement is None:
            requirement = SceneEventRequirement()

        issues = []
        suggestions = []

        # 1. 检查禁止模式（占位符）
        for pattern in requirement.forbidden_patterns:
            if re.search(pattern, event, re.IGNORECASE):
                issues.append(f"占位符事件: '{event}' 匹配禁止模式 '{pattern}'")
                suggestions.append("请用具体事件替代，如'发现XXX'或'获得XXX'")

        # 2. 检测动作
        has_action = EventStructureDetector.detect_action(event)
        if requirement.require_action and not has_action:
            issues.append("缺少动作词（动词）")
            suggestions.append("请添加明确的动作词，如'发现'、'获得'、'进入'等")

        # 3. 检测目标
        has_target, target_type = EventStructureDetector.detect_target(event)
        if requirement.require_target and not has_target:
            issues.append("缺少目标对象（名词）")
            suggestions.append("请添加具体的目标对象")
        elif has_target and target_type == "abstract":
            issues.append("目标对象较抽象，建议使用具体对象")
            suggestions.append("可考虑具体化目标，如'古墓'、'玉简'等")

        # 4. 检测状态转换
        has_transition, transition_type = EventStructureDetector.detect_transition(event)
        if requirement.require_transition and not has_transition:
            issues.append("缺少可验证的状态转换")
            suggestions.append("请包含转换词，如'发现'、'获得'、'进入'等")
        elif has_transition and transition_type == "implicit":
            issues.append("状态转换较模糊，建议明确化")
            suggestions.append("可使用'确认'、'揭示'等更明确的转换词")

        # 5. 确定状态
        if issues:
            has_severe = any(
                "占位符" in i or "缺少动作" in i or "缺少目标" in i or "缺少可验证" in i
                for i in issues
            )
            if has_severe:
                status = EventValidationStatus.INVALID
            else:
                status = EventValidationStatus.WARNING
        else:
            status = EventValidationStatus.VALID

        return EventValidationResult(
            event=event,
            status=status,
            has_action=has_action,
            has_target=has_target,
            target_type=target_type,
            has_transition=has_transition,
            transition_type=transition_type,
            issues=issues,
            suggestions=suggestions,
        )

    @classmethod
    def validate_scene(
        cls,
        events: List[str],
        requirement: Optional[SceneEventRequirement] = None,
    ) -> SceneValidationResult:
        if requirement is None:
            requirement = SceneEventRequirement()

        # ========== 空 events 特殊处理 ==========
        if not events:
            return SceneValidationResult(
                valid=True,
                events=[],
                valid_count=0,
                warning_count=1,
                invalid_count=0,
                contract_quality="empty",
                blocking_errors=0,
                summary="场景没有 must_events（作为警告，允许通过）",
            )

        results = [cls.validate_event(e, requirement) for e in events]
        total_count = len(results)
        valid_count = sum(1 for r in results if r.status == EventValidationStatus.VALID)
        warning_count = sum(1 for r in results if r.status == EventValidationStatus.WARNING)
        invalid_count = sum(1 for r in results if r.status == EventValidationStatus.INVALID)

        has_invalid = invalid_count > 0

        # 数量不足仅警告，不阻断
        if not has_invalid and total_count < requirement.minimum_count:
            for r in results:
                if r.status == EventValidationStatus.VALID:
                    r.issues.append(f"事件数量不足（{total_count} < {requirement.minimum_count}），但结构有效，作为警告")
            valid = True
            contract_quality = "partial" if total_count > 0 else "empty"
        else:
            valid = not has_invalid
            if has_invalid:
                contract_quality = "invalid"
            elif valid_count == total_count:
                contract_quality = "complete"
            elif valid_count > 0 or warning_count > 0:
                contract_quality = "partial"
            else:
                contract_quality = "empty"

        summary = f"通过: {valid_count} 有效, {warning_count} 警告, {invalid_count} 无效"
        if total_count < requirement.minimum_count and valid:
            summary += f"（数量不足 {total_count}/{requirement.minimum_count}，作为警告）"
        elif not valid:
            issues_summary = []
            if invalid_count > 0:
                issues_summary.append(f"存在 {invalid_count} 个无效事件")
            summary = f"不通过: {', '.join(issues_summary)}"

        return SceneValidationResult(
            valid=valid,
            events=results,
            valid_count=valid_count,
            warning_count=warning_count,
            invalid_count=invalid_count,
            contract_quality=contract_quality,
            blocking_errors=invalid_count,
            summary=summary,
        )

    @classmethod
    def get_invalid_events(cls, result: SceneValidationResult) -> List[str]:
        return [r.event for r in result.events if r.status == EventValidationStatus.INVALID]

    @classmethod
    def get_warning_events(cls, result: SceneValidationResult) -> List[str]:
        return [r.event for r in result.events if r.status == EventValidationStatus.WARNING]

    @classmethod
    def has_blocking_issues(cls, result: SceneValidationResult) -> bool:
        # 只根据 invalid_count 判断，不引入 contract_quality
        return result.invalid_count > 0