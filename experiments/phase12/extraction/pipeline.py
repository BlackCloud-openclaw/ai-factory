"""
ExtractionPipeline：统一编排提取流程
"""

import uuid
from pathlib import Path
from typing import Protocol, Optional

from .provider import FailureProvider
from .normalizer import FailureNormalizer
from .classifier import FailureClassifier
from .builder import CorpusSampleBuilder
from .validator import SchemaValidator
from .exporter import YamlExporter
from .repository import CorpusRepository
from .config import ExtractionConfig
from .result import MutableExtractionStats, ExtractionResult
from .models import ClassifiedFailure


class SampleIdGenerator(Protocol):
    """ID 生成器协议"""
    def generate(self, classified: ClassifiedFailure, index: int) -> str:
        ...


class SequentialSampleIdGenerator:
    """顺序 ID 生成器（基于 UUID5，避免碰撞）"""

    def generate(self, classified: ClassifiedFailure, index: int) -> str:
        namespace = uuid.NAMESPACE_DNS
        raw = f"{classified.normalized.source.value}:{classified.normalized.timestamp.isoformat()}:{index}"
        uid = str(uuid.uuid5(namespace, raw))[:8]
        return f"corpus.{classified.failure_mode.value}.{uid}"


class ExtractionPipeline:
    """统一提取流程编排器"""

    def __init__(
        self,
        provider: FailureProvider,
        normalizer: FailureNormalizer,
        classifier: FailureClassifier,
        builder: CorpusSampleBuilder,
        id_generator: SampleIdGenerator,
        repository: CorpusRepository,  # 必需
        validator: Optional[SchemaValidator] = None,
        exporter: Optional[YamlExporter] = None,
        config: Optional[ExtractionConfig] = None,
    ):
        self._provider = provider
        self._normalizer = normalizer
        self._classifier = classifier
        self._builder = builder
        self._id_generator = id_generator
        self._repository = repository
        self._validator = validator or SchemaValidator()
        self._exporter = exporter or YamlExporter()
        self._config = config or ExtractionConfig.default()

    def run(self) -> ExtractionResult:
        """执行完整提取流程"""
        stats = MutableExtractionStats()
        valid_samples = []

        for idx, raw in enumerate(self._provider.iter_records(), 1):
            stats.total_records += 1

            normalized = self._normalizer.normalize(raw)
            if not normalized:
                stats.skipped += 1
                continue
            stats.normalized += 1

            classified = self._classifier.classify(normalized)
            if not classified:
                stats.skipped += 1
                continue
            stats.classified += 1

            sample_id = self._id_generator.generate(classified, idx)

            sample = self._builder.build(classified, sample_id)
            if not sample:
                stats.skipped += 1
                continue
            stats.built += 1

            passed, errs = self._validator.validate(sample)
            if not passed:
                stats.invalid += 1
                stats.add_error(f"{sample_id}: {errs}")
                continue
            stats.validated += 1

            self._exporter.export(sample, self._repository)
            stats.exported += 1
            valid_samples.append(sample)

        if valid_samples:
            self._repository.add_samples(valid_samples, self._config.corpus_version)

        frozen_stats = stats.freeze()
        return ExtractionResult(
            stats=frozen_stats,
            samples=tuple(valid_samples),
        )