"""
CorpusSampleBuilder：从 ClassifiedFailure 构建 CorpusSample
"""

from typing import Optional, Dict, Any

from ..corpus.models import (
    CorpusSample,
    CorpusArtifacts,
    ExpectedResult,
    FailureMode,
    Difficulty,
    ExpectationType,
)
from .models import ClassifiedFailure


class CorpusSampleBuilder:
    """从 ClassifiedFailure 构建 CorpusSample（ID 由调用方提供）"""

    def __init__(self, expected_profiles: Dict[str, Dict[str, Any]]):
        self._expected_profiles = expected_profiles

    def build(self, classified: ClassifiedFailure, sample_id: str) -> Optional[CorpusSample]:
        """构建 CorpusSample，ID 必须由调用方提供"""
        normalized = classified.normalized
        category = classified.failure_mode

        if not self._has_valid_data(normalized):
            return None

        expected = self._build_expected(category)

        return CorpusSample(
            id=sample_id,
            version="1.0",
            category=category,
            failure_modes=(category,),
            difficulty=Difficulty.MEDIUM,
            language="zh-CN",
            scene_before=normalized.scene_text or "",
            scene_after=None,  # 不复制，避免 before == after
            draft_before=normalized.draft_before,
            draft_after=normalized.draft_after,
            expected=expected,
            artifacts=CorpusArtifacts(
                planning_contract=normalized.planning_contract,
                snapshot_before=normalized.snapshot_before,
                snapshot_after=normalized.snapshot_after,
                events=normalized.events,
                runtime_metrics=normalized.runtime_metrics,
            ),
            source=normalized.source.value,
            license="internal",
            tags=tuple(normalized.tags + (category.value,)),
        )

    def _has_valid_data(self, normalized) -> bool:
        return normalized.scene_text is not None and len(normalized.scene_text) > 50

    def _build_expected(self, category: FailureMode) -> Dict[str, ExpectedResult]:
        profile = self._expected_profiles.get(category.value, {})
        expected = {}
        for metric, spec in profile.items():
            expected[metric] = ExpectedResult(
                metric=metric,
                expectation_type=ExpectationType.from_string(spec.get("type", "range")),
                minimum=spec.get("min"),
                maximum=spec.get("max"),
                exact=spec.get("exact"),
                tolerance=spec.get("tolerance", 0.0),
            )
        return expected