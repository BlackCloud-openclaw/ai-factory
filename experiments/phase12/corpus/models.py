"""
Corpus 领域模型（修订版）
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence, Mapping, Tuple, List, Dict, Any
from types import MappingProxyType

from .failure_mode import FailureMode, Difficulty, ExpectationType


# ============================================================================
# ExpectedResult（含 tolerance，CUSTOM 抛异常）
# ============================================================================

@dataclass(frozen=True)
class ExpectedResult:
    """
    某个指标的期望结果。
    支持 exact、range、boolean、custom 四种类型。
    """
    metric: str
    expectation_type: ExpectationType
    exact: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    boolean: Optional[bool] = None
    custom: Optional[Dict[str, Any]] = None
    tolerance: float = 0.0  # 仅 Exact 类型使用

    def matches(self, score: Optional[float]) -> bool:
        """判断实际得分是否满足期望"""
        if score is None:
            return False

        if self.expectation_type == ExpectationType.EXACT:
            return abs(score - self.exact) <= self.tolerance if self.exact is not None else False
        elif self.expectation_type == ExpectationType.RANGE:
            if self.minimum is not None and score < self.minimum:
                return False
            if self.maximum is not None and score > self.maximum:
                return False
            return True
        elif self.expectation_type == ExpectationType.BOOLEAN:
            return score == (1.0 if self.boolean else 0.0)
        elif self.expectation_type == ExpectationType.CUSTOM:
            raise NotImplementedError(
                f"CUSTOM expectation type for metric '{self.metric}' is not implemented. "
                "Please provide a custom matcher or use a supported expectation type."
            )
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "type": self.expectation_type.value,
            "exact": self.exact,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "boolean": self.boolean,
            "custom": self.custom,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpectedResult":
        return cls(
            metric=data["metric"],
            expectation_type=ExpectationType.from_string(data["type"]),
            exact=data.get("exact"),
            minimum=data.get("minimum"),
            maximum=data.get("maximum"),
            boolean=data.get("boolean"),
            custom=data.get("custom"),
            tolerance=data.get("tolerance", 0.0),
        )


# ============================================================================
# CorpusArtifacts（使用通用 Mapping，而非具体 DTO）
# ============================================================================

@dataclass(frozen=True)
class CorpusArtifacts:
    """样本关联的可选运行时数据（通用序列化结构）"""
    planning_contract: Optional[Mapping[str, Any]] = None
    snapshot_before: Optional[Mapping[str, Any]] = None
    snapshot_after: Optional[Mapping[str, Any]] = None
    events: Optional[Sequence[Mapping[str, Any]]] = None
    runtime_metrics: Optional[Mapping[str, Any]] = None

    @property
    def has_valid_artifacts(self) -> bool:
        return any([
            self.planning_contract is not None,
            self.snapshot_before is not None,
            self.snapshot_after is not None,
            self.events is not None,
            self.runtime_metrics is not None,
        ])


# ============================================================================
# CorpusSample
# ============================================================================

@dataclass(frozen=True)
class CorpusSample:
    """单个 Corpus 样本的领域模型"""
    id: str
    version: str

    category: FailureMode
    failure_modes: Tuple[FailureMode, ...]
    difficulty: Difficulty
    language: str

    scene_before: str
    scene_after: str

    draft_before: Optional[str]
    draft_after: Optional[str]

    expected: Mapping[str, ExpectedResult]
    artifacts: CorpusArtifacts

    source: str
    license: str
    tags: Tuple[str, ...] = ()

    @property
    def is_revision_sample(self) -> bool:
        return self.draft_before is not None and self.draft_after is not None


# ============================================================================
# CorpusMetadata（使用 MappingProxyType 实现不可变）
# ============================================================================

@dataclass(frozen=True)
class CorpusMetadata:
    """Corpus 元数据"""
    version: str
    created_at: datetime
    categories: Tuple[str, ...]

    # 以下字段由 Loader 自动计算，不在 YAML 中维护
    total_samples: int = 0
    failure_mode_distribution: Mapping[str, int] = field(default_factory=MappingProxyType)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorpusMetadata":
        return cls(
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            categories=tuple(data.get("categories", [])),
        )

    @classmethod
    def compute(cls, samples: Sequence[CorpusSample], version: str, created_at: datetime, categories: Tuple[str, ...]) -> "CorpusMetadata":
        """从样本列表计算元数据"""
        distribution = {}
        for s in samples:
            for fm in s.failure_modes:
                distribution[fm.value] = distribution.get(fm.value, 0) + 1

        return cls(
            version=version,
            created_at=created_at,
            categories=categories,
            total_samples=len(samples),
            failure_mode_distribution=MappingProxyType(distribution),
        )


# ============================================================================
# Corpus（根对象，支持链式过滤）
# ============================================================================

@dataclass(frozen=True)
class Corpus:
    """Corpus 根对象，Loader 的唯一返回值"""
    metadata: CorpusMetadata
    samples: Tuple[CorpusSample, ...]

    def filter_by_failure_mode(self, mode: FailureMode) -> "Corpus":
        filtered = [s for s in self.samples if mode in s.failure_modes]
        return self._with_samples(filtered)

    def filter_by_difficulty(self, difficulty: Difficulty) -> "Corpus":
        filtered = [s for s in self.samples if s.difficulty == difficulty]
        return self._with_samples(filtered)

    def filter_by_tags(self, tags: set[str], mode: str = "any") -> "Corpus":
        """按标签过滤，支持 'any' 或 'all' 模式"""
        if mode == "any":
            filtered = [s for s in self.samples if any(t in s.tags for t in tags)]
        elif mode == "all":
            filtered = [s for s in self.samples if all(t in s.tags for t in tags)]
        else:
            raise ValueError(f"Unknown tag filter mode: {mode}")
        return self._with_samples(filtered)

    def filter_by_category(self, category: FailureMode) -> "Corpus":
        filtered = [s for s in self.samples if s.category == category]
        return self._with_samples(filtered)

    def query(self, **filters) -> "Corpus":
        """通用链式查询接口"""
        result = self
        for key, value in filters.items():
            if key == "failure_mode":
                result = result.filter_by_failure_mode(value)
            elif key == "difficulty":
                result = result.filter_by_difficulty(value)
            elif key == "category":
                result = result.filter_by_category(value)
            elif key == "tags":
                result = result.filter_by_tags(value, mode=filters.get("tag_mode", "any"))
        return result

    def _with_samples(self, samples: List[CorpusSample]) -> "Corpus":
        new_metadata = CorpusMetadata.compute(
            samples=samples,
            version=self.metadata.version,
            created_at=self.metadata.created_at,
            categories=self.metadata.categories,
        )
        return Corpus(
            metadata=new_metadata,
            samples=tuple(samples),
        )

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)