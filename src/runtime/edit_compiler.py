"""
EditCompiler - Phase 7B-3: Snapshot 驱动的修复计划生成
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

from src.runtime.observation_ir import ObservationIR, SentenceSpan
from src.runtime.validator import ComplianceReport, LayerComplianceResult
from src.runtime.snapshot import RuntimeSnapshot
from src.surfaces.definition import RepairStrategy
from src.capabilities import Repairs


class EditOperation(Enum):
    INSERT_AFTER = "insert_after"
    REPLACE_SENTENCE = "replace_sentence"
    INSERT_BEFORE = "insert_before"
    MERGE_INTO = "merge_into"


@dataclass
class EditAction:
    operation: EditOperation
    anchor_sentence_id: str
    payload_type: str
    reference_pattern_id: Optional[str] = None
    preserve_sentence_ids: List[str] = field(default_factory=list)
    preserve_pattern_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation.value,
            "anchor_sentence_id": self.anchor_sentence_id,
            "payload_type": self.payload_type,
            "reference_pattern_id": self.reference_pattern_id,
            "preserve_sentence_ids": self.preserve_sentence_ids,
            "preserve_pattern_ids": self.preserve_pattern_ids,
        }


@dataclass
class EditPlan:
    source_hash: str
    diagnosis_id: str
    actions: List[EditAction]
    target_layers: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_hash": self.source_hash,
            "diagnosis_id": self.diagnosis_id,
            "actions": [a.to_dict() for a in self.actions],
            "target_layers": self.target_layers,
        }


class EditCompiler:
    """
    EditCompiler 是 Revision IR 的唯一生产者。
    Phase 7B-3: 完全由 Snapshot 驱动，无 Surface 特判。
    """

    def __init__(self):
        self._operation_registry = {
            Repairs.INSERT_DIALOGUE: self._build_insert_dialogue_action,
            Repairs.INSERT_AFTER: self._build_insert_after_action,
            Repairs.REPLACE_SENTENCE: self._build_replace_sentence_action,
        }

    def compile_with_snapshot(
        self,
        snapshot: RuntimeSnapshot,
        report: ComplianceReport,
        text: str,
        ir: ObservationIR,
        diagnosis_id: str = "DEFAULT",
    ) -> EditPlan:
        """
        从 Snapshot 加载 Repair 策略，生成 EditPlan

        流程：
        1. 遍历 snapshot.surfaces
        2. 对每个 Surface，检查其 Repair 策略
        3. 如果对应层不合规，生成 EditAction
        """
        actions: List[EditAction] = []
        target_layers: List[str] = []

        # 构建层合规索引
        layer_compliance: Dict[str, bool] = {}
        for layer_result in report.layer_results:
            layer_compliance[layer_result.layer] = layer_result.compliant

        for surface in snapshot.surfaces:
            for strategy in surface.repair.repair_strategies:
                target_layer = strategy.target_layer
                trigger = strategy.trigger

                # 检查该层是否不合规
                is_compliant = layer_compliance.get(target_layer, True)
                if not is_compliant:
                    # 生成 EditAction
                    action = self._build_action(strategy, ir, text, snapshot)
                    if action:
                        actions.append(action)
                        target_layers.append(target_layer)

        return EditPlan(
            source_hash=ir.source_hash,
            diagnosis_id=diagnosis_id,
            actions=actions,
            target_layers=list(set(target_layers)),
        )

    def _build_action(
        self,
        strategy: RepairStrategy,
        ir: ObservationIR,
        text: str,
        snapshot: RuntimeSnapshot,
    ) -> Optional[EditAction]:
        """根据 RepairStrategy 构建 EditAction"""
        operation = strategy.operation
        builder = self._operation_registry.get(operation)
        if not builder:
            return None
        return builder(strategy, ir, text, snapshot)

    def _build_insert_dialogue_action(
        self,
        strategy: RepairStrategy,
        ir: ObservationIR,
        text: str,
        snapshot: RuntimeSnapshot,
    ) -> EditAction:
        """构建插入对话的 Action"""
        # 找到最后一个句子作为锚点
        anchor_sentence = self._find_last_sentence(ir)
        return EditAction(
            operation=EditOperation.INSERT_AFTER,
            anchor_sentence_id=anchor_sentence.id if anchor_sentence else "",
            payload_type=strategy.payload_type,
            preserve_sentence_ids=[],
            preserve_pattern_ids=[],
        )

    def _build_insert_after_action(
        self,
        strategy: RepairStrategy,
        ir: ObservationIR,
        text: str,
        snapshot: RuntimeSnapshot,
    ) -> EditAction:
        """构建通用插入 Action"""
        anchor_sentence = self._find_last_sentence(ir)
        return EditAction(
            operation=EditOperation.INSERT_AFTER,
            anchor_sentence_id=anchor_sentence.id if anchor_sentence else "",
            payload_type=strategy.payload_type,
            preserve_sentence_ids=[],
            preserve_pattern_ids=[],
        )

    def _build_replace_sentence_action(
        self,
        strategy: RepairStrategy,
        ir: ObservationIR,
        text: str,
        snapshot: RuntimeSnapshot,
    ) -> EditAction:
        """构建替换句子 Action（暂未实现）"""
        anchor_sentence = self._find_first_sentence(ir)
        return EditAction(
            operation=EditOperation.REPLACE_SENTENCE,
            anchor_sentence_id=anchor_sentence.id if anchor_sentence else "",
            payload_type=strategy.payload_type,
            preserve_sentence_ids=[],
            preserve_pattern_ids=[],
        )

    def _find_last_sentence(self, ir: ObservationIR) -> Optional[SentenceSpan]:
        """找到最后一句"""
        if ir.sentences:
            return ir.sentences[-1]
        return None

    def _find_first_sentence(self, ir: ObservationIR) -> Optional[SentenceSpan]:
        if ir.sentences:
            return ir.sentences[0]
        return None

    # ---------- 向后兼容（Phase 6）----------
    # 保留原有方法，但标记为 deprecated
    def compile(self, ir: ObservationIR, report: ComplianceReport, diagnosis_id: str = "DEFAULT") -> EditPlan:
        """
        Phase 6 兼容接口（已弃用）
        请使用 compile_with_snapshot
        """
        # 警告信息
        import warnings
        warnings.warn(
            "EditCompiler.compile(ir, report) is deprecated. Use compile_with_snapshot(snapshot, report, text, ir) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # 回退到空计划
        return EditPlan(
            source_hash=ir.source_hash,
            diagnosis_id=diagnosis_id,
            actions=[],
            target_layers=[],
        )