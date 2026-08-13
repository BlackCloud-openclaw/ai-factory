# src/writing/audit/preservation.py
"""
Phase 10.2.2: Preservation Analyzer — 基于 Lineage 的字段保留率分析
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from uuid import UUID

from .trace import ExecutionTrace
from .payload_resolver import PayloadResolver
from .field_extractor import FieldExtractor
from .field_comparator import FieldComparator, ComparisonResult, Existence, ChangeType
from .trace_traversal import TraceTraversal


@dataclass
class LineagePreservation:
    source: UUID
    sink: UUID
    lineage: List[UUID]
    statuses: Dict[UUID, ComparisonResult]
    end_retention_rate: float
    first_absent_artifact: Optional[UUID] = None

    @property
    def is_fully_preserved(self) -> bool:
        return self.end_retention_rate == 1.0 and self.first_absent_artifact is None

    @property
    def is_fully_lost(self) -> bool:
        return self.end_retention_rate == 0.0


@dataclass
class FieldPreservation:
    field_name: str
    sources: List[UUID]
    sinks: List[UUID]
    lineages: List[LineagePreservation]
    end_retention_rate: float
    path_retention_rate: float
    first_absent_artifact: Optional[UUID] = None

    @property
    def is_fully_preserved(self) -> bool:
        return self.end_retention_rate == 1.0

    @property
    def is_fully_lost(self) -> bool:
        return self.end_retention_rate == 0.0


@dataclass
class PreservationReport:
    execution_id: str
    total_fields: int
    fields: Dict[str, FieldPreservation]
    lost_fields: List[str] = field(default_factory=list)
    preserved_fields: List[str] = field(default_factory=list)
    partial_fields: List[str] = field(default_factory=list)

    @property
    def overall_retention_rate(self) -> float:
        if not self.fields:
            return 0.0
        total = sum(f.end_retention_rate for f in self.fields.values())
        return total / len(self.fields)

    def to_markdown(self) -> str:
        lines = [
            "# Preservation Analysis Report",
            "",
            f"**Execution ID:** `{self.execution_id}`",
            f"**Total Fields:** {self.total_fields}",
            f"**Overall End Retention Rate:** {self.overall_retention_rate:.2%}",
            "",
            "## Summary",
            "",
            f"- ✅ Fully Preserved: {len(self.preserved_fields)}",
            f"- ⚠️ Partially Preserved: {len(self.partial_fields)}",
            f"- ❌ Fully Lost: {len(self.lost_fields)}",
            "",
            "## Field Details",
            "",
        ]
        for name, f in self.fields.items():
            status = "✅" if f.is_fully_preserved else "⚠️" if not f.is_fully_lost else "❌"
            lines.append(f"### {status} `{name}`")
            lines.append(f"- End Retention Rate: {f.end_retention_rate:.2%}")
            lines.append(f"- Path Retention Rate: {f.path_retention_rate:.2%}")
            if f.first_absent_artifact:
                lines.append(f"- First Absent: `{f.first_absent_artifact}`")
            lines.append(f"- Sources: {len(f.sources)}, Sinks: {len(f.sinks)}")
            lines.append("")
            for lp in f.lineages:
                lines.append(f"  - Lineage {lp.source}->{lp.sink}: end_rate={lp.end_retention_rate:.2%}, first_absent={lp.first_absent_artifact}")
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "total_fields": self.total_fields,
            "overall_retention_rate": self.overall_retention_rate,
            "lost_fields": self.lost_fields,
            "preserved_fields": self.preserved_fields,
            "partial_fields": self.partial_fields,
            "fields": {
                name: {
                    "end_retention_rate": f.end_retention_rate,
                    "path_retention_rate": f.path_retention_rate,
                    "sources": [str(s) for s in f.sources],
                    "sinks": [str(s) for s in f.sinks],
                    "lineages": [
                        {
                            "source": str(lp.source),
                            "sink": str(lp.sink),
                            "lineage": [str(a) for a in lp.lineage],
                            "end_retention_rate": lp.end_retention_rate,
                            "first_absent": str(lp.first_absent_artifact) if lp.first_absent_artifact else None,
                        }
                        for lp in f.lineages
                    ],
                }
                for name, f in self.fields.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PreservationReport":
        return cls(
            execution_id=data.get("execution_id", ""),
            total_fields=data.get("total_fields", 0),
            fields={},
            lost_fields=data.get("lost_fields", []),
            preserved_fields=data.get("preserved_fields", []),
            partial_fields=data.get("partial_fields", []),
        )


class PreservationAnalyzer:
    def __init__(
        self,
        resolver: PayloadResolver,
        fields: Optional[List[str]] = None,
        extractor: Optional[FieldExtractor] = None,
    ):
        self._resolver = resolver
        self._fields = fields or ["goal", "conflict", "outcome", "must_events", "characters", "constraints", "scene_spec"]
        self._extractor = extractor or FieldExtractor()

    def analyze(self, trace: ExecutionTrace) -> PreservationReport:
        traversal = TraceTraversal(trace)
        sources = traversal.get_sources()
        sinks = traversal.get_sinks()
        field_preservations = {}

        for field in self._fields:
            source_values = {}
            for sid in sources:
                artifact = traversal.get_artifact(sid)
                if artifact is None:
                    continue
                try:
                    payload = self._resolver.resolve(artifact.payload_ref)
                    result = self._extractor.extract(payload, field)
                    if result.found:
                        source_values[sid] = result.value
                except ValueError:
                    continue

            if not source_values:
                continue

            lineages_pres = []
            for sid, src_val in source_values.items():
                for sink in sinks:
                    paths = traversal.get_lineages_to_sink(sid, sink)
                    if not paths:
                        continue
                    for path in paths:
                        if len(path) < 2:
                            continue
                        statuses = {}
                        prev_value = src_val
                        first_absent = None
                        for aid in path[1:]:
                            artifact = traversal.get_artifact(aid)
                            if artifact is None:
                                continue
                            try:
                                payload = self._resolver.resolve(artifact.payload_ref)
                                result = self._extractor.extract(payload, field)
                                if result.found:
                                    comp = FieldComparator.compare(prev_value, result.value)
                                    statuses[aid] = comp
                                    prev_value = result.value
                                    if comp.existence == Existence.REMOVED and first_absent is None:
                                        first_absent = aid
                                else:
                                    comp = ComparisonResult(
                                        existence=Existence.REMOVED,
                                        change=ChangeType.UNCHANGED,
                                        retention_ratio=0.0,
                                    )
                                    statuses[aid] = comp
                                    if first_absent is None:
                                        first_absent = aid
                                    prev_value = None
                            except ValueError:
                                statuses[aid] = ComparisonResult(
                                    existence=Existence.UNKNOWN,
                                    change=ChangeType.UNCHANGED,
                                    retention_ratio=0.0,
                                )
                        if not statuses:
                            continue
                        last_aid = path[-1]
                        end_retention = statuses.get(last_aid, ComparisonResult(
                            existence=Existence.REMOVED,
                            change=ChangeType.UNCHANGED,
                            retention_ratio=0.0,
                        )).retention_ratio if last_aid in statuses else 0.0
                        lp = LineagePreservation(
                            source=sid,
                            sink=sink,
                            lineage=path,
                            statuses=statuses,
                            end_retention_rate=end_retention,
                            first_absent_artifact=first_absent,
                        )
                        lineages_pres.append(lp)

            if not lineages_pres:
                continue

            avg_end = sum(lp.end_retention_rate for lp in lineages_pres) / len(lineages_pres)
            avg_path = sum(
                sum(r.retention_ratio for r in lp.statuses.values()) / max(1, len(lp.statuses))
                for lp in lineages_pres
            ) / len(lineages_pres)
            first_absent = None
            for lp in lineages_pres:
                if lp.first_absent_artifact is not None:
                    first_absent = lp.first_absent_artifact
                    break

            fp = FieldPreservation(
                field_name=field,
                sources=list(source_values.keys()),
                sinks=sinks,
                lineages=lineages_pres,
                end_retention_rate=avg_end,
                path_retention_rate=avg_path,
                first_absent_artifact=first_absent,
            )
            field_preservations[field] = fp

        lost, preserved, partial = [], [], []
        for name, fp in field_preservations.items():
            if fp.is_fully_preserved:
                preserved.append(name)
            elif fp.is_fully_lost:
                lost.append(name)
            else:
                partial.append(name)

        return PreservationReport(
            execution_id=str(trace.execution_id),
            total_fields=len(self._fields),
            fields=field_preservations,
            lost_fields=lost,
            preserved_fields=preserved,
            partial_fields=partial,
        )