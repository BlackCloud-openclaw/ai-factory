# src/narrative/realizers/reference.py

import logging
from typing import Optional

from src.narrative.artifact import NarrativeArtifact
from src.narrative.context import NarrativeContext
from src.narrative.constraint import NarrativeConstraint
from src.narrative.intent import ResolutionPlan
from src.narrative.realizer import NarrativeRealizer
from src.narrative.realizers.prompts import build_editor_prompt
from src.narrative.realizers.interfaces import TextGenerator

logger = logging.getLogger(__name__)


class ReferenceNarrativeRealizer:
    def __init__(
        self,
        text_generator: TextGenerator,
        constraint_formatter: Optional[callable] = None,
    ):
        self._text_generator = text_generator
        self._constraint_formatter = constraint_formatter or self._default_format_constraint

    def _default_format_constraint(self, constraint: NarrativeConstraint) -> str:
        return f"- 约束 ID: {constraint.constraint_id}"

    async def realize(
        self,
        artifact: NarrativeArtifact,
        context: NarrativeContext,
        plan: ResolutionPlan,
        constraint: NarrativeConstraint,
    ) -> NarrativeArtifact:
        if not plan.primary_intents:
            logger.info("No primary intents in plan, returning original artifact")
            return artifact

        constraint_summary = self._constraint_formatter(constraint)

        # 从上下文中提取决议文本
        resolution_text = None
        if context.resolution_context and context.resolution_context.resolutions:
            resolution_text = "\n".join(
                r.to_prompt() for r in context.resolution_context.resolutions
            )

        prompt = build_editor_prompt(
            artifact_text=artifact.text,
            context=context,
            intents=list(plan.primary_intents),
            constraint_summary=constraint_summary,
            resolution_text=resolution_text,
        )

        try:
            edited_text = await self._text_generator.generate(prompt)
            edited_text = self._normalize_output(edited_text)

            if not edited_text:
                logger.warning("Generated text is empty, returning original")
                return artifact

            return NarrativeArtifact(
                text=edited_text,
                artifact_id=artifact.artifact_id,
            )

        except Exception as e:
            logger.error(f"ReferenceNarrativeRealizer failed: {e}")
            return artifact

    def _normalize_output(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```") and "```" in text[3:]:
            lines = text.split("\n")
            start, end = 0, len(lines)
            for i, line in enumerate(lines):
                if line.strip().startswith("```"):
                    start = i + 1
                    break
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip().startswith("```"):
                    end = i
                    break
            text = "\n".join(lines[start:end]).strip()
        return text