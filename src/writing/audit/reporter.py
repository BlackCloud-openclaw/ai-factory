# src/writing/audit/reporter.py
"""
Phase 10.2.6: Reporter — 综合报告生成器
"""

from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Optional, Any

from .preservation import PreservationReport
from .attribution import AttributionReport
from .budget import BudgetReport
from .priority import PriorityReport, PriorityLevel


@dataclass(frozen=True)
class ComprehensiveReport:
    """
    综合报告：聚合所有分析结果。
    """
    execution_id: str
    preservation: PreservationReport
    attribution: AttributionReport
    budget: BudgetReport
    priority: PriorityReport

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "total_fields": self.preservation.total_fields,
            "fields_with_loss": len(self.preservation.lost_fields) + len(self.preservation.partial_fields),
            "fields_fully_preserved": len(self.preservation.preserved_fields),
            "total_targets": self.priority.total_targets,
            "top_priority": self._get_top_priority_summary(),
        }

    def _get_top_priority_summary(self) -> Optional[dict[str, Any]]:
        top = self.priority.top_critical or (self.priority.targets[0] if self.priority.targets else None)
        if top is None:
            return None
        return {
            "field": top.field_name,
            "stage": top.lost_stage,
            "severity": top.severity.value,
            "score": top.priority_score,
        }

    def to_markdown(self) -> str:
        sections = [
            self._render_header(),
            self._render_summary(),
            self._render_preservation(),
            self._render_attribution(),
            self._render_budget(),
            self._render_priority(),
        ]
        return "\n\n".join(sections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "summary": self.summary,
            "preservation": self.preservation.to_dict(),
            "attribution": self.attribution.to_dict(),
            "budget": self.budget.to_dict(),
            "priority": self.priority.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ComprehensiveReport":
        from .preservation import PreservationReport
        from .attribution import AttributionReport
        from .budget import BudgetReport
        from .priority import PriorityReport

        return cls(
            execution_id=data["execution_id"],
            preservation=PreservationReport.from_dict(data.get("preservation", {})),
            attribution=AttributionReport.from_dict(data.get("attribution", {})),
            budget=BudgetReport.from_dict(data.get("budget", {})),
            priority=PriorityReport.from_dict(data.get("priority", {})),
        )

    def to_console(self) -> str:
        lines = [
            f"📊 Report: {self.execution_id}",
            "",
            f"  Fields: {self.preservation.total_fields} total, "
            f"{self.preservation.overall_retention_rate:.1%} retention",
            f"  Priority: {self.priority.total_targets} targets",
        ]
        top = self._get_top_priority_summary()
        if top:
            lines.append(f"  🔥 Top: {top['field']} at {top['stage']} ({top['severity']})")
        return "\n".join(lines)

    # ========== 内部渲染方法 ==========

    def _render_header(self) -> str:
        return f"# Comprehensive Audit Report\n\n**Execution ID:** `{self.execution_id}`"

    def _render_summary(self) -> str:
        summary = self.summary
        lines = [
            "## Summary",
            "",
            f"- **Total Fields Analyzed:** {summary['total_fields']}",
            f"- **Fields with Loss:** {summary['fields_with_loss']}",
            f"- **Fully Preserved Fields:** {summary['fields_fully_preserved']}",
            f"- **Optimization Targets:** {summary['total_targets']}",
        ]
        top = summary.get("top_priority")
        if top:
            lines.append(f"- **Top Priority:** `{top['field']}` at `{top['stage']}` ({top['severity']}, score={top['score']:.1f})")
        return "\n".join(lines)

    def _render_preservation(self) -> str:
        lines = ["## 1. Preservation Analysis", ""]
        report = self.preservation

        if report.lost_fields:
            lines.append("### ❌ Fully Lost Fields")
            lines.append("")
            for f in report.lost_fields:
                fp = report.fields.get(f)
                if fp:
                    lines.append(f"- `{f}`: {fp.end_retention_rate:.1%} retained, first absent at `{fp.first_absent_artifact}`")
            lines.append("")

        if report.partial_fields:
            lines.append("### ⚠️ Partially Preserved Fields")
            lines.append("")
            for f in report.partial_fields:
                fp = report.fields.get(f)
                if fp:
                    lines.append(f"- `{f}`: {fp.end_retention_rate:.1%} retained")
            lines.append("")

        if report.preserved_fields:
            lines.append("### ✅ Fully Preserved Fields")
            lines.append("")
            lines.append(", ".join(f"`{f}`" for f in report.preserved_fields[:5]))
            if len(report.preserved_fields) > 5:
                lines.append(f"\n... and {len(report.preserved_fields) - 5} more")
            lines.append("")

        lines.append(f"\n**Overall Retention Rate:** {report.overall_retention_rate:.1%}")
        return "\n".join(lines)

    def _render_attribution(self) -> str:
        lines = ["## 2. Attribution Analysis", ""]
        report = self.attribution

        if not report.attributions:
            lines.append("No attribution data available.")
            return "\n".join(lines)

        lines.append("| Field | Lost Stage | Type |")
        lines.append("|-------|------------|------|")
        for field, attr in report.attributions.items():
            lines.append(f"| `{field}` | `{attr.lost_stage}` | {attr.attribution_type.value} |")
        return "\n".join(lines)

    def _render_budget(self) -> str:
        lines = ["## 3. Budget Analysis", ""]
        report = self.budget

        if not report.stages:
            lines.append("No budget data available.")
            return "\n".join(lines)

        lines.append(f"**Metric:** `{report.metric}`")
        lines.append(f"**Total Value:** {report.total_metric_value:,}")
        lines.append("")
        lines.append("| Stage | Value | Percentage | Score |")
        lines.append("|-------|-------|------------|-------|")
        for s in report.stages:
            lines.append(f"| {s.stage} | {s.metric_value:,} | {s.percentage:.1%} | {s.score:.2f} |")

        if report.anomalies:
            lines.append("")
            lines.append("### Anomalies")
            for a in report.anomalies:
                lines.append(f"- [{a.severity.value}] {a.message}")
        return "\n".join(lines)

    def _render_priority(self) -> str:
        lines = ["## 4. Priority Recommendations", ""]
        report = self.priority

        if report.total_targets == 0:
            lines.append("No optimization targets identified.")
            return "\n".join(lines)

        lines.append("| Priority | Field | Stage | Retention | Score |")
        lines.append("|----------|-------|-------|-----------|-------|")
        for t in report.targets[:10]:
            lines.append(f"| {t.severity.value} | `{t.field_name}` | `{t.lost_stage}` | {t.current_retention:.1%} | {t.priority_score:.1f} |")

        if report.total_targets > 10:
            lines.append(f"\n... and {report.total_targets - 10} more targets")

        if report.targets:
            lines.append("")
            lines.append("### Top Target Detail")
            top = report.targets[0]
            lines.append(f"**Field:** `{top.field_name}`")
            lines.append(f"**Stage:** `{top.lost_stage}`")
            lines.append(f"**Current Retention:** {top.current_retention:.1%}")
            lines.append(f"**Stage Score:** {top.stage_score:.2f}")
            lines.append("")
            lines.append("**Factor Breakdown:**")
            for f in top.factors:
                lines.append(f"- {f.name}: {f.score:.2f} × {f.weight:.2f} = {f.contribution:.2f}")

        return "\n".join(lines)


class Reporter:
    def generate(
        self,
        preservation_report: PreservationReport,
        attribution_report: AttributionReport,
        budget_report: BudgetReport,
        priority_report: PriorityReport,
    ) -> ComprehensiveReport:
        execution_id = preservation_report.execution_id
        if attribution_report.execution_id != execution_id:
            raise ValueError("execution_id mismatch between preservation and attribution reports")
        if budget_report.execution_id != execution_id:
            raise ValueError("execution_id mismatch between preservation and budget reports")
        if priority_report.execution_id != execution_id:
            raise ValueError("execution_id mismatch between preservation and priority reports")

        return ComprehensiveReport(
            execution_id=execution_id,
            preservation=preservation_report,
            attribution=attribution_report,
            budget=budget_report,
            priority=priority_report,
        )