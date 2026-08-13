"""
CorpusLoader：使用显式 Manifest，支持 Schema 版本校验
"""

import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
from datetime import datetime

from .models import (
    Corpus,
    CorpusMetadata,
    CorpusSample,
    CorpusArtifacts,
    ExpectedResult,
)
from .failure_mode import FailureMode, Difficulty, ExpectationType


class CorpusLoader:
    """加载 Corpus，使用显式 Manifest，不涉及 Runtime"""

    SUPPORTED_VERSIONS = {"1.0", "1.1"}

    def load(self, path: Union[str, Path]) -> Corpus:
        """加载 Corpus Manifest"""
        path = Path(path)
        if path.is_file():
            index_path = path
        else:
            index_path = path / "corpus.yaml"
            if not index_path.exists():
                raise FileNotFoundError(f"corpus.yaml not found in {path}")

        with open(index_path, "r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f)

        version = manifest.get("version")
        if version not in self.SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported corpus version: {version}. "
                f"Supported: {self.SUPPORTED_VERSIONS}"
            )

        base_dir = index_path.parent
        created_at = datetime.fromisoformat(manifest.get("created_at", datetime.now().isoformat()))
        categories = tuple(manifest.get("categories", []))

        samples: List[CorpusSample] = []
        seen_ids = set()

        for entry in manifest.get("samples", []):
            sample_path = base_dir / entry["path"]
            if not sample_path.exists():
                raise FileNotFoundError(f"Sample not found: {sample_path}")

            sample = self.load_sample(sample_path)

            if sample.id in seen_ids:
                raise ValueError(f"Duplicate sample ID: {sample.id}")
            seen_ids.add(sample.id)
            samples.append(sample)

        metadata = CorpusMetadata.compute(
            samples=samples,
            version=version,
            created_at=created_at,
            categories=categories,
        )

        return Corpus(metadata=metadata, samples=tuple(samples))

    def load_sample(self, path: Union[str, Path]) -> CorpusSample:
        """加载单个样本文件"""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return self._parse_sample(data)

    def _parse_sample(self, data: Dict[str, Any]) -> CorpusSample:
        sample_id = data["id"]
        version = data["version"]

        category = FailureMode.from_string(data["category"])
        failure_modes = tuple(
            FailureMode.from_string(m) for m in data.get("failure_modes", [])
        )
        difficulty = Difficulty.from_string(data["difficulty"])
        language = data.get("language", "zh-CN")

        scene_before = data.get("scene_before", "")
        scene_after = data.get("scene_after", "")

        draft_before = data.get("draft_before")
        draft_after = data.get("draft_after")

        expected = self._parse_expected(data.get("expected", {}))
        artifacts = self._parse_artifacts(data.get("artifacts", {}))

        source = data.get("source", "unknown")
        license_ = data.get("license", "internal")
        tags = tuple(data.get("tags", []))

        return CorpusSample(
            id=sample_id,
            version=version,
            category=category,
            failure_modes=failure_modes,
            difficulty=difficulty,
            language=language,
            scene_before=scene_before,
            scene_after=scene_after,
            draft_before=draft_before,
            draft_after=draft_after,
            expected=expected,
            artifacts=artifacts,
            source=source,
            license=license_,
            tags=tags,
        )

    def _parse_expected(self, data: Dict[str, Any]) -> Dict[str, ExpectedResult]:
        result = {}
        for metric, spec in data.items():
            normalized_spec = {
                "metric": metric,
                "type": spec.get("type"),
                "exact": spec.get("exact"),
                "minimum": spec.get("min"),
                "maximum": spec.get("max"),
                "boolean": spec.get("boolean"),
                "custom": spec.get("custom"),
                "tolerance": spec.get("tolerance", 0.0),
            }
            result[metric] = ExpectedResult.from_dict(normalized_spec)
        return result

    def _parse_artifacts(self, data: Dict[str, Any]) -> CorpusArtifacts:
        return CorpusArtifacts(
            planning_contract=data.get("planning_contract"),
            snapshot_before=data.get("snapshot_before"),
            snapshot_after=data.get("snapshot_after"),
            events=data.get("events"),
            runtime_metrics=data.get("runtime_metrics"),
        )


# ========== 兼容旧 API（支持 str 或 Path） ==========
def load_samples(path: Union[str, Path]):
    """兼容旧 API：加载 Corpus 样本列表"""
    loader = CorpusLoader()
    corpus = loader.load(path)
    return list(corpus.samples)


def load_corpus(path: Union[str, Path]):
    """兼容旧 API：加载 Corpus 对象"""
    loader = CorpusLoader()
    return loader.load(path)