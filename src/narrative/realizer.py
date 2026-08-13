# src/narrative/realizer.py

from typing import Protocol, runtime_checkable

from src.narrative.artifact import NarrativeArtifact
from src.narrative.context import NarrativeContext
from src.narrative.constraint import NarrativeConstraint
from src.narrative.intent import ResolutionPlan


@runtime_checkable
class NarrativeRealizer(Protocol):
    async def realize(
        self,
        artifact: NarrativeArtifact,
        context: NarrativeContext,
        plan: ResolutionPlan,
        constraint: NarrativeConstraint,
    ) -> NarrativeArtifact:
        ...