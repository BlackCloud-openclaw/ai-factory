"""
JudgeContextFactory：从 CorpusSample 构建 JudgeContext
"""

from typing import Optional

from .models import CorpusSample
from ..judge.context import JudgeContext


class JudgeContextFactory:
    """独立构建 JudgeContext，与 ContextFactory 解耦"""

    def create(self, sample: CorpusSample) -> JudgeContext:
        return JudgeContext(
            previous_scene_text=sample.scene_before,
            character_summary=self._extract_character_summary(sample),
            world_summary=self._extract_world_summary(sample),
        )

    def _extract_character_summary(self, sample: CorpusSample) -> Optional[str]:
        if sample.artifacts.snapshot_before:
            chars = sample.artifacts.snapshot_before.get("characters", {})
            if chars:
                return "\n".join([
                    f"{name}: realm={info.get('realm', '?')}, hp={info.get('hp', '?')}"
                    for name, info in chars.items()
                ])
        return None

    def _extract_world_summary(self, sample: CorpusSample) -> Optional[str]:
        return None