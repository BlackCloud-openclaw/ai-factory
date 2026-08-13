# src/narrative/loops/quality_loop.py

import logging
from dataclasses import dataclass
from typing import Optional

from src.narrative.artifact import NarrativeArtifact
from src.narrative.context import NarrativeContext
from src.narrative.constraint import NarrativeConstraint
from src.narrative.intent import (
    ResolutionPlan,
    SatisfactionReport,
    IntentSatisfaction,
    NarrativeIntentSet,
)
from src.narrative.realizer import NarrativeRealizer

logger = logging.getLogger(__name__)


@dataclass
class LoopResult:
    artifact: NarrativeArtifact
    plan: ResolutionPlan
    report: SatisfactionReport
    iterations: int
    accepted: bool
    max_iterations_reached: bool = False


class QualityLoop:
    def __init__(
        self,
        realizer: NarrativeRealizer,
        satisfaction: Optional[IntentSatisfaction] = None,
        max_iterations: int = 3,
        acceptance_threshold: float = 0.7,
    ):
        self._realizer = realizer
        self._satisfaction = satisfaction or IntentSatisfaction()
        self._max_iterations = max_iterations
        self._acceptance_threshold = acceptance_threshold

    async def run(
        self,
        artifact: NarrativeArtifact,
        context: NarrativeContext,
        plan: ResolutionPlan,
        constraint: NarrativeConstraint,
    ) -> LoopResult:
        if not plan.primary_intents:
            return LoopResult(
                artifact=artifact,
                plan=plan,
                report=SatisfactionReport(overall=1.0, passed=True),
                iterations=0,
                accepted=True,
            )

        current = artifact
        iterations = 0
        intents = NarrativeIntentSet(intents=plan.primary_intents)

        for i in range(self._max_iterations):
            iterations = i + 1

            current = await self._realizer.realize(
                current,
                context,
                plan,
                constraint,
            )

            report = await self._satisfaction.evaluate(current, intents)

            if report.passed and report.overall >= self._acceptance_threshold:
                logger.info(f"Quality Loop accepted after {iterations} iterations")
                return LoopResult(
                    artifact=current,
                    plan=plan,
                    report=report,
                    iterations=iterations,
                    accepted=True,
                )

            logger.info(
                f"Iteration {iterations}/{self._max_iterations}: "
                f"satisfaction={report.overall:.2f}"
            )

        final_report = await self._satisfaction.evaluate(current, intents)
        return LoopResult(
            artifact=current,
            plan=plan,
            report=final_report,
            iterations=iterations,
            accepted=False,
            max_iterations_reached=True,
        )