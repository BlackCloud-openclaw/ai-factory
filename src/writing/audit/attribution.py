# src/writing/audit/attribution.py
"""
Phase 10.2.3: Attribution Analyzer — 定位第一次丢失发生的 Artifact Edge
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from uuid import UUID
from enum import Enum

from .trace import ExecutionTrace, StageRecord
from .preservation import FieldPreservation, LineagePreservation, PreservationReport
from .payload_resolver import PayloadResolver
from .field_extractor import FieldExtractor
from .field_comparator import ComparisonResult, Existence, ChangeType


class AttributionType(Enum):
    INPUT_LOST = "input_lost"
    OUTPUT_LOST = "output_lost"
    TRANSFORM_LOST = "transform_lost"
    UNKNOWN = "unknown"


@dataclass
class AttributionResult:
    field_name: str
    lost_artifact_id: UUID
    input_artifact_id: UUID
    edge_index: int
    lost_stage: str
    attribution_type: AttributionType
    input_existence: Existence
    output_existence: Existence
    reason: str = ""
    possible_causes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "lost_artifact_id": str(self.lost_artifact_id),
            "input_artifact_id": str(self.input_artifact_id),
            "edge_index": self.edge_index,
            "lost_stage": self.lost_stage,
            "attribution_type": self.attribution_type.value,
            "input_existence": self.input_existence.value,
            "output_existence": self.output_existence.value,
            "reason": self.reason,
            "possible_causes": self.possible_causes,
        }


@dataclass
class AttributionReport:
    execution_id: str
    total_fields_analyzed: int
    fields_with_loss: int
    fields_without_loss: int
    attributions: Dict[str, AttributionResult]
    by_type: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "total_fields_analyzed": self.total_fields_analyzed,
            "fields_with_loss": self.fields_with_loss,
            "fields_without_loss": self.fields_without_loss,
            "by_type": self.by_type,
            "attributions": {k: v.to_dict() for k, v in self.attributions.items()},
        }

    def to_markdown(self) -> str:
        lines = [
            "# Attribution Analysis Report",
            "",
            f"**Execution ID:** `{self.execution_id}`",
            f"**Fields with Loss:** {self.fields_with_loss}",
            f"**Fields without Loss:** {self.fields_without_loss}",
            "",
            "## Summary by Type",
            "",
        ]
        for t, count in self.by_type.items():
            lines.append(f"- {t}: {count}")
        lines.append("")

        lines.append("## Field Details")
        lines.append("")
        for field_name, attr in self.attributions.items():
            lines.append(f"### `{field_name}`")
            lines.append(f"- Lost Artifact: `{attr.lost_artifact_id}`")
            lines.append(f"- Input Artifact: `{attr.input_artifact_id}`")
            lines.append(f"- Edge Index: {attr.edge_index}")
            lines.append(f"- Stage: `{attr.lost_stage}`")
            lines.append(f"- Type: `{attr.attribution_type.value}`")
            lines.append(f"- Input Existence: `{attr.input_existence.value}`")
            lines.append(f"- Output Existence: `{attr.output_existence.value}`")
            if attr.reason:
                lines.append(f"- Reason: {attr.reason}")
            if attr.possible_causes:
                lines.append("- Possible Causes:")
                for cause in attr.possible_causes:
                    lines.append(f"  - {cause}")
            lines.append("")
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict) -> "AttributionReport":
        return cls(
            execution_id=data.get("execution_id", ""),
            total_fields_analyzed=data.get("total_fields_analyzed", 0),
            fields_with_loss=data.get("fields_with_loss", 0),
            fields_without_loss=data.get("fields_without_loss", 0),
            attributions={},
            by_type=data.get("by_type", {}),
        )


class AttributionAnalyzer:
    def __init__(
        self,
        resolver: PayloadResolver,
        extractor: Optional[FieldExtractor] = None,
    ):
        self._resolver = resolver
        self._extractor = extractor or FieldExtractor()

    def analyze(
        self,
        trace: ExecutionTrace,
        preservation_report: PreservationReport,
    ) -> AttributionReport:
        attributions = {}
        fields_with_loss = 0
        fields_without_loss = 0
        by_type = {}

        for field_name, field_pres in preservation_report.fields.items():
            if field_pres.is_fully_preserved:
                fields_without_loss += 1
                continue

            fields_with_loss += 1
            result = self._attribut_field(trace, field_name, field_pres)
            attributions[field_name] = result
            by_type[result.attribution_type.value] = by_type.get(result.attribution_type.value, 0) + 1

        return AttributionReport(
            execution_id=str(trace.execution_id),
            total_fields_analyzed=preservation_report.total_fields,
            fields_with_loss=fields_with_loss,
            fields_without_loss=fields_without_loss,
            attributions=attributions,
            by_type=by_type,
        )

    def _attribut_field(
        self,
        trace: ExecutionTrace,
        field_name: str,
        field_pres: FieldPreservation,
    ) -> AttributionResult:
        best_lp = None
        best_index = None
        best_aid = None
        best_parent = None

        for lp in field_pres.lineages:
            lineage = lp.lineage
            for i in range(1, len(lineage)):
                aid = lineage[i]
                comp = lp.statuses.get(aid)
                if comp is None:
                    continue
                if comp.existence == Existence.REMOVED:
                    if best_index is None or i < best_index:
                        best_index = i
                        best_aid = aid
                        best_parent = lineage[i-1] if i > 0 else None
                        best_lp = lp
                    break

        if best_aid is None:
            for lp in field_pres.lineages:
                for i in range(1, len(lp.lineage)):
                    aid = lp.lineage[i]
                    comp = lp.statuses.get(aid)
                    if comp is None:
                        continue
                    if comp.existence != Existence.PRESENT:
                        if best_index is None or i < best_index:
                            best_index = i
                            best_aid = aid
                            best_parent = lp.lineage[i-1] if i > 0 else None
                            best_lp = lp
                        break

        if best_aid is None:
            return AttributionResult(
                field_name=field_name,
                lost_artifact_id=UUID("00000000-0000-0000-0000-000000000000"),
                input_artifact_id=UUID("00000000-0000-0000-0000-000000000000"),
                edge_index=-1,
                lost_stage="unknown",
                attribution_type=AttributionType.UNKNOWN,
                input_existence=Existence.UNKNOWN,
                output_existence=Existence.UNKNOWN,
                reason="无法定位丢失边",
            )

        stage, role = self._find_artifact_stage(trace, best_aid)
        stage_name = stage.stage if stage else "unknown"

        if best_parent in field_pres.sources:
            parent_comp = ComparisonResult(
                existence=Existence.PRESENT,
                change=ChangeType.UNCHANGED,
                retention_ratio=1.0,
            )
        else:
            parent_comp = best_lp.statuses.get(best_parent)

        if parent_comp and parent_comp.existence == Existence.PRESENT:
            attr_type = AttributionType.TRANSFORM_LOST
            reason = f"字段在 Artifact {best_aid} 处理过程中丢失"
            causes = ["转换逻辑未包含该字段", "输出格式改变", "字段被过滤"]
        else:
            attr_type = AttributionType.INPUT_LOST
            reason = f"字段在 Artifact {best_aid} 的输入中已丢失"
            causes = ["上游未输出", "格式不兼容", "被过滤"]

        return AttributionResult(
            field_name=field_name,
            lost_artifact_id=best_aid,
            input_artifact_id=best_parent if best_parent else UUID("00000000-0000-0000-0000-000000000000"),
            edge_index=best_index,
            lost_stage=stage_name,
            attribution_type=attr_type,
            input_existence=parent_comp.existence if parent_comp else Existence.UNKNOWN,
            output_existence=Existence.REMOVED,
            reason=reason,
            possible_causes=causes,
        )

    def _find_artifact_stage(
        self,
        trace: ExecutionTrace,
        artifact_id: UUID,
    ) -> tuple[Optional[StageRecord], str]:
        for stage in trace.stages:
            if artifact_id in stage.output_artifacts:
                return stage, "output"
            if artifact_id in stage.input_artifacts:
                return stage, "input"
        return None, ""