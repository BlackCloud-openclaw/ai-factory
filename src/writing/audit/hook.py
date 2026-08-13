# src/writing/audit/hook.py
"""
Phase 10.3.2: RuntimeHook — 自动捕获 Writer Runtime 执行
"""

import inspect
import logging
from functools import wraps
from typing import Optional, Callable, TypeVar, ParamSpec, Any

from .coordinator import AuditCoordinator, AuditConfig
from .trace import PayloadRef
from .payload_resolver import PayloadResolver, MemoryPayloadResolver

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def audit_writer(
    resolver: Optional[PayloadResolver] = None,
    config: Optional[AuditConfig] = None,
):
    """
    装饰器：自动审计 Writer 执行。
    """
    if resolver is None:
        resolver = MemoryPayloadResolver()

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                coordinator = AuditCoordinator(resolver=resolver, config=config)
                novel_id, volume, chapter, scene_idx = _extract_params(args, kwargs)
                with coordinator.audit(novel_id, volume, chapter, scene_idx) as ctx:
                    result = await func(*args, **kwargs)
                    if ctx.collector and result is not None:
                        ref = PayloadRef(f"memory://writer/{id(result)}")
                        resolver.register(ref, result)
                        ctx.collector.record_reference(
                            artifact_type="writer_result",
                            payload_ref=ref,
                            digest="",
                            size_bytes=0,
                        )
                        ctx.collector.record_stage("writer", outputs={"result_ref": str(ref)})
                    return result
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                coordinator = AuditCoordinator(resolver=resolver, config=config)
                novel_id, volume, chapter, scene_idx = _extract_params(args, kwargs)
                with coordinator.audit(novel_id, volume, chapter, scene_idx) as ctx:
                    result = func(*args, **kwargs)
                    if ctx.collector and result is not None:
                        ref = PayloadRef(f"memory://writer/{id(result)}")
                        resolver.register(ref, result)
                        ctx.collector.record_reference(
                            artifact_type="writer_result",
                            payload_ref=ref,
                            digest="",
                            size_bytes=0,
                        )
                        ctx.collector.record_stage("writer", outputs={"result_ref": str(ref)})
                    return result
            return sync_wrapper
    return decorator


def _extract_params(args: tuple, kwargs: dict) -> tuple[str, int, int, int]:
    novel_id = kwargs.get("novel_id", "unknown")
    volume = kwargs.get("volume", 1)
    chapter = kwargs.get("chapter", 1)
    scene_idx = kwargs.get("scene_idx", 0)

    if novel_id == "unknown" and args:
        first_arg = args[0]
        if hasattr(first_arg, "novel_id"):
            novel_id = getattr(first_arg, "novel_id", "unknown")
        elif hasattr(first_arg, "get") and callable(getattr(first_arg, "get", None)):
            novel_id = first_arg.get("novel_id", "unknown")
    return novel_id, volume, chapter, scene_idx


class AuditHook:
    """
    手动 Hook：占位实现，未来可扩展。
    """
    def __init__(self, coordinator: AuditCoordinator):
        self._coordinator = coordinator

    def record_planning(self, planning_result: dict) -> None:
        self._record_stage("planning", planning_result)

    def record_observation(self, observation_result: dict) -> None:
        self._record_stage("observation", observation_result)

    def record_prompt(self, prompt_result: dict) -> None:
        self._record_stage("prompt", prompt_result)

    def record_draft(self, draft_result: dict) -> None:
        self._record_stage("draft", draft_result)

    def record_coverage(self, coverage_result: dict) -> None:
        self._record_stage("coverage", coverage_result)

    def _record_stage(self, stage: str, data: dict) -> None:
        logger.debug(f"AuditHook.record_{stage} called")