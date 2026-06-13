from pydantic import BaseModel
from typing import Any, Optional, List
from datetime import datetime

class BenchmarkResult(BaseModel):
    query_id: str
    value: Any

class ChapterBenchmark(BaseModel):
    chapter_key: str
    results: List[BenchmarkResult]

class BenchmarkMetadata(BaseModel):
    schema_version: int = 1
    novel_id: str
    generated_at: str
    git_commit: Optional[str] = None
    generator_version: str = "benchmark_v1"

class BenchmarkBaseline(BaseModel):
    metadata: BenchmarkMetadata
    chapters: dict[str, ChapterBenchmark]

class ProjectionAuditResult(BaseModel):
    projection_match: bool
    current_hash: str
    rebuilt_hash: str
    missing_predicates: List[str] = []
    extra_predicates: List[str] = []
    checked_predicate_count: int = 0
    metadata: dict[str, Any] = {}

class CoverageReport(BaseModel):
    source_type: str
    total_fields: int
    mapped_fields: int
    coverage: float
    missing_fields: List[str] = []
    children: List["CoverageReport"] = []
