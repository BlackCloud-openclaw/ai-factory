"""
YamlExporter：导出 CorpusSample 为 YAML
"""

import yaml
from pathlib import Path

from ..corpus.models import CorpusSample
from .repository import CorpusRepository


class YamlExporter:
    """导出 YAML，路径由 Repository 提供"""

    def export(self, sample: CorpusSample, repository: CorpusRepository) -> Path:
        """导出单个样本"""
        filepath = repository.sample_path(sample)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = self._to_dict(sample)
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

        return filepath

    def _to_dict(self, sample: CorpusSample) -> dict:
        data = {
            "id": sample.id,
            "version": sample.version,
            "category": sample.category.value,
            "failure_modes": [fm.value for fm in sample.failure_modes],
            "difficulty": sample.difficulty.value,
            "language": sample.language,
            "scene_before": sample.scene_before,
            "scene_after": sample.scene_after,
            "source": sample.source,
            "license": sample.license,
            "tags": list(sample.tags),
            "expected": self._expected_to_dict(sample.expected),
            "artifacts": {
                "planning_contract": sample.artifacts.planning_contract or None,
                "snapshot_before": sample.artifacts.snapshot_before or None,
                "snapshot_after": sample.artifacts.snapshot_after or None,
                "events": sample.artifacts.events or None,
                "runtime_metrics": sample.artifacts.runtime_metrics or None,
            },
        }
        if sample.draft_before:
            data["draft_before"] = sample.draft_before
        if sample.draft_after:
            data["draft_after"] = sample.draft_after
        return {k: v for k, v in data.items() if v is not None and v != {} and v != []}

    def _expected_to_dict(self, expected):
        result = {}
        for metric, exp in expected.items():
            if exp.expectation_type.value in ("range", "exact"):
                result[metric] = {
                    "type": exp.expectation_type.value,
                    "min": exp.minimum,
                    "max": exp.maximum,
                    "exact": exp.exact,
                    "tolerance": exp.tolerance,
                }
        return result