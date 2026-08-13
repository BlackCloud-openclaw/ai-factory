# src/narrative/resolution/context_builder.py

from typing import Optional
import dataclasses

from src.narrative.intent import ResolutionPlan
from src.narrative.context import NarrativeContext, ResolutionContext


def build_resolution_context(plan: ResolutionPlan) -> Optional[ResolutionContext]:
    """从 ResolutionPlan 构建 ResolutionContext，若无决议则返回 None"""
    if not plan.resolutions:
        return None
    return ResolutionContext(
        conflicts=plan.conflicts,
        resolutions=plan.resolutions,
    )


def enrich_narrative_context(
    context: NarrativeContext,
    plan: ResolutionPlan,
) -> NarrativeContext:
    """
    如果 plan 包含决议，则创建包含 resolution_context 的新上下文副本。
    若无决议，返回原 context。
    """
    resolution_ctx = build_resolution_context(plan)
    if resolution_ctx is None:
        return context
    return dataclasses.replace(context, resolution_context=resolution_ctx)