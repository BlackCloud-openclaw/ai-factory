# src/writing/audit/__init__.py

from .registry import (
    StageRegistry,
    ArtifactTypeRegistry,
    StageDefinition,
    ArtifactTypeDefinition,
    create_default_stage_registry,
    create_default_artifact_type_registry,
)
from .payload_ref import PayloadRef
from .trace import ExecutionTrace, StageRecord, Artifact, SCHEMA_VERSION
from .collector import TraceCollector
from .payload_resolver import PayloadResolver, MemoryPayloadResolver
from .field_extractor import FieldExtractor, ExtractionResult
from .field_comparator import FieldComparator, ComparisonResult, Existence, ChangeType
from .trace_traversal import TraceTraversal
from .preservation import (
    PreservationAnalyzer,
    PreservationReport,
    FieldPreservation,
    LineagePreservation,
)
from .attribution import (
    AttributionAnalyzer,
    AttributionReport,
    AttributionResult,
    AttributionType,
)
from .budget import (
    BudgetAnalyzer,
    MetricBudgetAnalyzer,
    BudgetReport,
    StageMetricBudget,
    BudgetAnomaly,
    BudgetSeverity,
    BudgetAnomalyKind,
)
from .priority import (
    PriorityEngine,
    PriorityReport,
    OptimizationTarget,
    PriorityFactor,
    PriorityLevel,
    PriorityPolicy,
)
from .reporter import Reporter, ComprehensiveReport
from .coordinator import AuditCoordinator, AuditConfig, AuditContext
from .hook import AuditHook, audit_writer
from .store import AuditReportStore, ReportEntry

__all__ = [
    "StageRegistry",
    "ArtifactTypeRegistry",
    "StageDefinition",
    "ArtifactTypeDefinition",
    "create_default_stage_registry",
    "create_default_artifact_type_registry",
    "PayloadRef",
    "ExecutionTrace",
    "StageRecord",
    "Artifact",
    "SCHEMA_VERSION",
    "TraceCollector",
    "PayloadResolver",
    "MemoryPayloadResolver",
    "FieldExtractor",
    "ExtractionResult",
    "FieldComparator",
    "ComparisonResult",
    "Existence",
    "ChangeType",
    "TraceTraversal",
    "PreservationAnalyzer",
    "PreservationReport",
    "FieldPreservation",
    "LineagePreservation",
    "AttributionAnalyzer",
    "AttributionReport",
    "AttributionResult",
    "AttributionType",
    "BudgetAnalyzer",
    "MetricBudgetAnalyzer",
    "BudgetReport",
    "StageMetricBudget",
    "BudgetAnomaly",
    "BudgetSeverity",
    "BudgetAnomalyKind",
    "PriorityEngine",
    "PriorityReport",
    "OptimizationTarget",
    "PriorityFactor",
    "PriorityLevel",
    "PriorityPolicy",
    "Reporter",
    "ComprehensiveReport",
    "AuditCoordinator",
    "AuditConfig",
    "AuditContext",
    "AuditHook",
    "audit_writer",
    "AuditReportStore",
    "ReportEntry",
]