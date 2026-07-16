# src/runtime/failure_analyzer.py
"""
Failure Analyzer - Runtime 执行诊断器

职责：
1. 接收 ComplianceReport 和 Draft 文本
2. 分析每一层失败的具体原因
3. 输出 FailureAnalysis 诊断 IR（结构化，不包含修复方案）

设计原则：
- 纯诊断：只分析失败原因，不生成修复方案
- IR 驱动：输出 FailureAnalysis 作为诊断 IR
- 可扩展：每种失败类型可单独实现诊断逻辑
- 与 Validator 解耦：Validator 判断合规性，Analyzer 诊断原因
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import re
import uuid

from src.runtime.validator import ComplianceReport, LayerCompliance, ObservationLevel
from src.runtime.compiler import LayerControlTargets, LayerTarget


# ============================================================
# 失败类型分类（Failure Taxonomy）
# ============================================================

class FailureType(Enum):
    """失败类型分类"""
    # Reasoning Layer 失败
    NO_STATE = "no_state"                           # State 关键词完全未出现
    STATE_MENTIONED_ONLY = "state_mentioned_only"   # State 出现但未进入推理
    STATE_WRONG_LAYER = "state_wrong_layer"         # State 出现在 Construction/Emotion，但未进入 Reasoning（层泄漏）
    STATE_NOT_BOUND_TO_REASONING = "state_not_bound_to_reasoning"  # State 与推理标记共现失败
    STATE_CONTRADICTS_POLICY = "state_contradicts_policy"         # State 改变事件但被 Policy 禁止
    REASONING_INSUFFICIENT = "reasoning_insufficient"             # 推理存在但不充分
    
    # Justification Layer 失败
    JUSTIFICATION_NO_STATE = "justification_no_state"             # State 未出现在决策理由中
    JUSTIFICATION_MENTIONED_ONLY = "justification_mentioned_only" # State 提及但未成为理由
    JUSTIFICATION_WEAK = "justification_weak"                     # State 成为理由但不够有力
    
    # Construction Layer 失败
    CONSTRUCTION_NO_STATE = "construction_no_state"               # State 未影响叙事实现
    CONSTRUCTION_WEAK = "construction_weak"                       # State 影响叙事但力度不足
    
    # Prediction Layer 失败
    PREDICTION_CHANGED = "prediction_changed"                     # 事件选择被改变但 Policy 要求固定
    PREDICTION_UNCLEAR = "prediction_unclear"                     # 事件选择不明确
    
    # 通用
    VALIDATOR_UNCERTAIN = "validator_uncertain"                   # Validator 无法确定


# ============================================================
# 诊断严重程度
# ============================================================

class Severity(Enum):
    """诊断严重程度"""
    CRITICAL = "critical"      # 必须修复，否则控制完全失效
    HIGH = "high"              # 需要修复，否则控制显著受损
    MEDIUM = "medium"          # 建议修复，有一定影响
    LOW = "low"                # 可选修复，影响有限


# ============================================================
# 诊断建议策略
# ============================================================

class SuggestedStrategy(Enum):
    """建议的修复策略（建议，非最终Patch）"""
    INSERT = "insert"          # 插入缺失内容
    REWRITE = "rewrite"        # 重写特定段落
    REPLACE = "replace"        # 替换整个层内容
    REJECT = "reject"          # 拒绝重试（全文重新生成）
    NONE = "none"              # 无需修复


# ============================================================
# 诊断结果（Runtime Diagnosis IR）
# ============================================================

@dataclass(frozen=True)
class FailureAnalysis:
    """
    诊断分析结果 - Runtime 诊断 IR
    
    这是 FailureAnalyzer 的唯一输出，包含诊断事实。
    不包含任何修复方案或计划。
    """
    id: str                              # 唯一标识，如 "F001"
    layer: str                           # "prediction" | "reasoning" | "justification" | "construction"
    failure_type: FailureType            # 失败类型
    severity: Severity                   # 严重程度
    confidence: float                    # 0-1，诊断置信度
    evidence: List[str] = field(default_factory=list)  # 支撑证据
    root_cause: str = ""                 # 人类可读的根本原因解释
    suggested_strategy: SuggestedStrategy = SuggestedStrategy.NONE  # 建议策略
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "layer": self.layer,
            "failure_type": self.failure_type.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "root_cause": self.root_cause,
            "suggested_strategy": self.suggested_strategy.value,
        }


@dataclass
class FailureDiagnosis:
    """
    完整的诊断结果（可能包含多个 FailureAnalysis）
    """
    analyses: List[FailureAnalysis]
    overall_confidence: float
    requires_attention: bool
    
    def failed_layers(self) -> List[str]:
        return [a.layer for a in self.analyses]
    
    def to_dict(self) -> dict:
        return {
            "analyses": [a.to_dict() for a in self.analyses],
            "overall_confidence": self.overall_confidence,
            "requires_attention": self.requires_attention,
        }


# ============================================================
# Failure Analyzer 主类
# ============================================================

class FailureAnalyzer:
    """
    执行失败诊断器 v1.0 - 规则驱动
    
    版本：1.0
    策略：基于 ComplianceReport + 文本分析
    适用范围：Phase 6.3A 实验验证
    """
    
    VERSION = "1.0"
    
    # State 关键词映射（与 Validator 保持一致）
    STATE_KEYWORDS = {
        "密信": ["密信", "留信", "信中", "信上", "信封"],
        "秘密": ["秘密", "真相", "线索", "隐情", "背后"],
        "主使": ["主使", "高层", "幕后", "指使"],
        "探索": ["探索", "调查", "追查", "探查"],
        "任务": ["任务", "师命", "复命", "职责"],
    }
    
    # 推理标记
    REASONING_MARKERS = [
        "因为", "所以", "于是", "既然", "为了", "思考", "意识到", "明白", 
        "觉得", "认为", "想到", "回想起", "认定", "推测", "推断", "判断", 
        "领悟", "察觉", "发现", "怀疑"
    ]
    
    # 决策动词
    DECISION_VERBS = ["决定", "选择", "权衡", "犹豫", "考虑", "放弃", "坚持"]
    
    def analyze(
        self,
        report: ComplianceReport,
        draft: str,
        targets: LayerControlTargets,
    ) -> FailureDiagnosis:
        """
        分析执行失败原因
        
        Args:
            report: Validator 输出的 ComplianceReport
            draft: Writer 生成的初稿文本
            targets: LayerControlTargets (IR)
            
        Returns:
            FailureDiagnosis: 诊断结果
        """
        analyses = []
        
        # 检查各层失败
        if not report.prediction.compliant:
            analyses.append(self._analyze_prediction_failure(report.prediction, draft, targets))
        
        if not report.reasoning.compliant:
            analyses.append(self._analyze_reasoning_failure(report.reasoning, draft, targets))
        
        if not report.justification.compliant:
            analyses.append(self._analyze_justification_failure(report.justification, draft, targets))
        
        if not report.construction.compliant:
            analyses.append(self._analyze_construction_failure(report.construction, draft, targets))
        
        # 如果没有失败，返回空诊断
        if not analyses:
            return FailureDiagnosis(
                analyses=[],
                overall_confidence=1.0,
                requires_attention=False,
            )
        
        # 计算整体置信度
        overall_conf = sum(a.confidence for a in analyses) / len(analyses)
        
        return FailureDiagnosis(
            analyses=analyses,
            overall_confidence=overall_conf,
            requires_attention=True,
        )
    
    # ============================================================
    # 各层失败诊断
    # ============================================================
    
    def _analyze_prediction_failure(
        self,
        compliance: LayerCompliance,
        draft: str,
        targets: LayerControlTargets,
    ) -> FailureAnalysis:
        """分析 Prediction Layer 失败"""
        
        # 提取事件选择
        choices = self._detect_choice(draft)
        
        # 判断失败类型
        if targets.prediction == LayerTarget.FIXED:
            # 预期固定事件，但出现了多个选项或模糊
            if len(choices) > 1:
                return FailureAnalysis(
                    id=f"F{_next_id():03d}",
                    layer="prediction",
                    failure_type=FailureType.PREDICTION_CHANGED,
                    severity=Severity.CRITICAL,
                    confidence=0.95,
                    evidence=[
                        f"检测到多个事件选项：{', '.join(choices)}",
                        f"Policy 要求 FIXED（事件锁定）",
                    ],
                    root_cause="Policy 要求事件锁定，但 Draft 中存在多个事件选项，违反了 L1 控制目标",
                    suggested_strategy=SuggestedStrategy.REJECT,
                )
            elif len(choices) == 0:
                return FailureAnalysis(
                    id=f"F{_next_id():03d}",
                    layer="prediction",
                    failure_type=FailureType.PREDICTION_UNCLEAR,
                    severity=Severity.HIGH,
                    confidence=0.85,
                    evidence=[
                        "未检测到明确的事件选择",
                    ],
                    root_cause="Policy 要求事件锁定，但 Draft 中未形成明确的事件选择",
                    suggested_strategy=SuggestedStrategy.REWRITE,
                )
        
        # 其他情况
        return FailureAnalysis(
            id=f"F{_next_id():03d}",
            layer="prediction",
            failure_type=FailureType.PREDICTION_UNCLEAR,
            severity=Severity.MEDIUM,
            confidence=0.70,
            evidence=[
                f"observed={compliance.observed.value}",
                "事件选择不明确或不符合 Policy",
            ],
            root_cause=f"Prediction Layer 观察为 {compliance.observed.value}，但与 Policy 要求不匹配",
            suggested_strategy=SuggestedStrategy.REWRITE,
        )
    
    def _analyze_reasoning_failure(
        self,
        compliance: LayerCompliance,
        draft: str,
        targets: LayerControlTargets,
    ) -> FailureAnalysis:
        """分析 Reasoning Layer 失败"""
        
        observed = compliance.observed
        state_keywords = self._extract_state_keywords(draft)
        
        # 规则驱动诊断
        if observed == ObservationLevel.MISSING:
            return self._diagnose_missing_reasoning(draft, state_keywords, targets)
        
        elif observed == ObservationLevel.MENTIONED:
            return self._diagnose_mentioned_reasoning(draft, state_keywords, targets)
        
        elif observed == ObservationLevel.INTEGRATED:
            # 如果 reported 为 INTEGRATED 但 compliant=False，说明分数不够
            return FailureAnalysis(
                id=f"F{_next_id():03d}",
                layer="reasoning",
                failure_type=FailureType.REASONING_INSUFFICIENT,
                severity=Severity.MEDIUM,
                confidence=0.75,
                evidence=[
                    f"State 已进入推理但合规分数不足（{compliance.score:.2f}）",
                    f"期望目标：{compliance.expected.value}",
                ],
                root_cause="State 进入了推理，但未达到 ENHANCED 或 NORMAL 要求的充分性标准",
                suggested_strategy=SuggestedStrategy.REWRITE,
            )
        
        # 默认
        return FailureAnalysis(
            id=f"F{_next_id():03d}",
            layer="reasoning",
            failure_type=FailureType.VALIDATOR_UNCERTAIN,
            severity=Severity.MEDIUM,
            confidence=0.50,
            evidence=[f"observed={compliance.observed.value}，无法确定具体失败原因"],
            root_cause="Validator 无法确定 Reasoning 层的具体失败原因",
            suggested_strategy=SuggestedStrategy.REWRITE,
        )
    
    def _analyze_justification_failure(
        self,
        compliance: LayerCompliance,
        draft: str,
        targets: LayerControlTargets,
    ) -> FailureAnalysis:
        """分析 Justification Layer 失败"""
        
        state_keywords = self._extract_state_keywords(draft)
        found = [kw for kw in state_keywords if kw in draft]
        
        if not found:
            return FailureAnalysis(
                id=f"F{_next_id():03d}",
                layer="justification",
                failure_type=FailureType.JUSTIFICATION_NO_STATE,
                severity=Severity.HIGH,
                confidence=0.95,
                evidence=["State 关键词未出现在决策理由中"],
                root_cause="决策理由中未引用 State，导致 Justification 层控制失效",
                suggested_strategy=SuggestedStrategy.INSERT,
            )
        
        if compliance.observed == ObservationLevel.MENTIONED:
            return FailureAnalysis(
                id=f"F{_next_id():03d}",
                layer="justification",
                failure_type=FailureType.JUSTIFICATION_MENTIONED_ONLY,
                severity=Severity.MEDIUM,
                confidence=0.80,
                evidence=[
                    f"State 出现但未成为决策理由（observed=Mentioned）",
                ],
                root_cause="State 被提及但未进入决策理由，Justification 层执行不充分",
                suggested_strategy=SuggestedStrategy.REWRITE,
            )
        
        return FailureAnalysis(
            id=f"F{_next_id():03d}",
            layer="justification",
            failure_type=FailureType.JUSTIFICATION_WEAK,
            severity=Severity.LOW,
            confidence=0.70,
            evidence=[
                f"observed={compliance.observed.value}，Justification 力度不足",
            ],
            root_cause=f"Justification 层执行力度不足（{compliance.observed.value}），可能原因：State 未充分融入决策理由",
            suggested_strategy=SuggestedStrategy.REWRITE,
        )
    
    def _analyze_construction_failure(
        self,
        compliance: LayerCompliance,
        draft: str,
        targets: LayerControlTargets,
    ) -> FailureAnalysis:
        """分析 Construction Layer 失败"""
        
        state_keywords = self._extract_state_keywords(draft)
        found = [kw for kw in state_keywords if kw in draft]
        
        if not found:
            return FailureAnalysis(
                id=f"F{_next_id():03d}",
                layer="construction",
                failure_type=FailureType.CONSTRUCTION_NO_STATE,
                severity=Severity.MEDIUM,
                confidence=0.90,
                evidence=["State 关键词未出现在叙事实现中"],
                root_cause="叙事实现中未引用 State，Construction 层控制失效",
                suggested_strategy=SuggestedStrategy.INSERT,
            )
        
        return FailureAnalysis(
            id=f"F{_next_id():03d}",
            layer="construction",
            failure_type=FailureType.CONSTRUCTION_WEAK,
            severity=Severity.LOW,
            confidence=0.70,
            evidence=[
                f"observed={compliance.observed.value}，Construction 力度不足",
            ],
            root_cause=f"Construction 层执行力度不足（{compliance.observed.value}），State 未充分影响叙事实现",
            suggested_strategy=SuggestedStrategy.REWRITE,
        )
    
    # ============================================================
    # 诊断辅助方法
    # ============================================================
    
    def _diagnose_missing_reasoning(
        self,
        draft: str,
        state_keywords: List[str],
        targets: LayerControlTargets,
    ) -> FailureAnalysis:
        """诊断 MISSING 类型的 Reasoning 失败"""
        
        # 检查 State 是否在文本中完全缺失
        if not state_keywords:
            return FailureAnalysis(
                id=f"F{_next_id():03d}",
                layer="reasoning",
                failure_type=FailureType.NO_STATE,
                severity=Severity.CRITICAL,
                confidence=0.98,
                evidence=[
                    "State 关键词完全未出现在文本中",
                    "当前 State 内容可能与场景文本不匹配",
                ],
                root_cause="State 关键词未出现在 Draft 中，Writer 可能完全忽略了 State 信息",
                suggested_strategy=SuggestedStrategy.INSERT,
            )
        
        # 检查 State 是否只出现在其他层（Construction 等）
        action_sentences = self._extract_action_sentences(draft)
        reasoning_blocks = self._extract_reasoning_blocks(draft)
        state_in_action = any(
            any(kw in sent for kw in state_keywords)
            for sent in action_sentences
        )
        state_in_reasoning = any(
            any(kw in block for kw in state_keywords)
            for block in reasoning_blocks
        )
        
        if state_in_action and not state_in_reasoning:
            return FailureAnalysis(
                id=f"F{_next_id():03d}",
                layer="reasoning",
                failure_type=FailureType.STATE_WRONG_LAYER,
                severity=Severity.HIGH,
                confidence=0.85,
                evidence=[
                    f"State 出现在动作/叙事描写中，但未出现在推理块中",
                    f"检测到 State 关键词：{', '.join(state_keywords[:3])}",
                ],
                root_cause="State 被泄漏到 Construction 层，但未进入 Reasoning 层（层泄漏）",
                suggested_strategy=SuggestedStrategy.REWRITE,
            )
        
        # State 出现了，但未在推理块中
        return FailureAnalysis(
            id=f"F{_next_id():03d}",
            layer="reasoning",
            failure_type=FailureType.STATE_NOT_BOUND_TO_REASONING,
            severity=Severity.HIGH,
            confidence=0.80,
            evidence=[
                f"State 关键词出现在文本中：{', '.join(state_keywords[:3])}",
                "但未与推理标记（因为、所以、思考等）共现",
            ],
            root_cause="State 出现在文本中，但未与推理结构绑定，无法进入 Reasoning 层",
            suggested_strategy=SuggestedStrategy.REWRITE,
        )
    
    def _diagnose_mentioned_reasoning(
        self,
        draft: str,
        state_keywords: List[str],
        targets: LayerControlTargets,
    ) -> FailureAnalysis:
        """诊断 MENTIONED 类型的 Reasoning 失败"""
        
        # 检查是否有推理结构但 State 未进入
        reasoning_blocks = self._extract_reasoning_blocks(draft)
        state_in_reasoning = any(
            any(kw in block for kw in state_keywords)
            for block in reasoning_blocks
        )
        
        # 检查是否与 Policy 冲突
        if targets.prediction == LayerTarget.FIXED:
            # 检查 State 是否改变了事件选择
            choices = self._detect_choice(draft)
            if len(choices) > 1:
                return FailureAnalysis(
                    id=f"F{_next_id():03d}",
                    layer="reasoning",
                    failure_type=FailureType.STATE_CONTRADICTS_POLICY,
                    severity=Severity.CRITICAL,
                    confidence=0.90,
                    evidence=[
                        f"State 影响了事件选择（检测到多个选项：{', '.join(choices)}）",
                        "但 Policy 要求 FIXED（事件锁定）",
                    ],
                    root_cause="State 进入推理并试图改变事件选择，但 Policy 禁止改变 L1 Prediction",
                    suggested_strategy=SuggestedStrategy.REJECT,
                )
        
        # 推理存在但 State 未充分绑定
        return FailureAnalysis(
            id=f"F{_next_id():03d}",
            layer="reasoning",
            failure_type=FailureType.STATE_MENTIONED_ONLY,
            severity=Severity.MEDIUM,
            confidence=0.75,
            evidence=[
                f"State 被提及但未充分参与推理",
                f"期望目标：{targets.reasoning.value}",
            ],
            root_cause="State 在文本中被提及，但未充分进入推理结构，无法满足 ENHANCED 要求",
            suggested_strategy=SuggestedStrategy.REWRITE,
        )
    
    # ============================================================
    # 文本分析辅助方法（与 Validator 保持一致）
    # ============================================================
    
    def _extract_state_keywords(self, text: str) -> List[str]:
        """从文本中提取活跃的 State 关键词"""
        found = []
        for group, keywords in self.STATE_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    found.append(kw)
        return found
    
    def _detect_choice(self, text: str) -> List[str]:
        """检测文本中的事件选择倾向"""
        choices = []
        choice_indicators = {
            "宗门": "宗门",
            "返回": "宗门",
            "师命": "宗门",
            "禁地": "禁地",
            "探查": "禁地",
            "标记": "标记",
            "等待": "标记",
            "观察": "观察",
            "高处": "观察",
        }
        for indicator, label in choice_indicators.items():
            if indicator in text:
                choices.append(label)
        return list(set(choices))
    
    def _extract_reasoning_blocks(self, text: str) -> List[str]:
        """提取包含推理标记的句子块"""
        sentences = re.split(r'[。！？；\n]', text)
        blocks = []
        for sent in sentences:
            if any(marker in sent for marker in self.REASONING_MARKERS):
                blocks.append(sent)
        return blocks
    
    def _extract_action_sentences(self, text: str) -> List[str]:
        """提取包含动作描写的句子"""
        action_verbs = ["走", "踏", "迈", "拿", "放", "握", "看", "听", "说", "站", "坐", "行", "停", "动", "冲", "挡", "砍"]
        sentences = re.split(r'[。！？；\n]', text)
        action_sents = []
        for sent in sentences:
            if any(verb in sent for verb in action_verbs):
                action_sents.append(sent)
        return action_sents


# ============================================================
# 便捷函数
# ============================================================

_counter = 0

def _next_id() -> int:
    """生成简单的递增 ID"""
    global _counter
    _counter += 1
    return _counter


def analyze_failures(
    report: ComplianceReport,
    draft: str,
    targets: LayerControlTargets,
) -> FailureDiagnosis:
    """便捷函数：分析执行失败"""
    analyzer = FailureAnalyzer()
    return analyzer.analyze(report, draft, targets)