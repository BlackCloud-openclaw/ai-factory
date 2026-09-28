# src/writing/__init__.py
"""
Writer Runtime Module — 公共 API 导出
"""

# 延迟导入，避免循环依赖
def __getattr__(name):
    if name == "WorldState":
        from .world_state import WorldState
        return WorldState
    elif name == "StateDelta":
        from .delta import StateDelta
        return StateDelta
    elif name == "EventType":
        from .events import EventType
        return EventType
    elif name == "NarrativeEvent":
        from .events import NarrativeEvent
        return NarrativeEvent
    elif name == "NarrativeEventStore":
        from .event_store import NarrativeEventStore
        return NarrativeEventStore
    elif name == "SnapshotManager":
        from .snapshot_manager import SnapshotManager
        return SnapshotManager
    elif name == "CompressedState":
        from .memory_hierarchy import CompressedState
        return CompressedState
    elif name == "Predicate":
        from .causality.predicate import Predicate
        return Predicate
    elif name == "PredicateDelta":
        from .causality.delta import PredicateDelta
        return PredicateDelta
    elif name == "Projector":
        from .causality.projector import Projector
        return Projector
    elif name == "PlanningContract":
        from .planning_contract import PlanningContract
        return PlanningContract
    elif name == "ContextCompiler":
        from .context_compiler import ContextCompiler
        return ContextCompiler
    elif name == "ControlledWriter":
        from .controlled_writer import ControlledWriter
        return ControlledWriter
    elif name == "SceneCompletionService":
        from .services.scene_completion import SceneCompletionService
        return SceneCompletionService
    elif name == "ScenePlanningService":
        from .services.scene_planning import ScenePlanningService
        return ScenePlanningService
    elif name == "WritingService":
        from .services.writing import WritingService
        return WritingService
    elif name == "ChapterTransitionService":
        from .services.chapter_transition import ChapterTransitionService
        return ChapterTransitionService
    elif name == "VersionedWriter":
        from .services.versioned_writer import VersionedWriter
        return VersionedWriter
    elif name == "VersionedWritingResult":
        from .services.versioned_writer import VersionedWritingResult
        return VersionedWritingResult
    elif name == "ContractSanityGuard":
        from .contract_sanity import ContractSanityGuard
        return ContractSanityGuard
    elif name == "SanityResult":
        from .contract_sanity import SanityResult
        return SanityResult
    elif name == "SanityViolation":
        from .contract_sanity import SanityViolation
        return SanityViolation
    elif name == "DefaultRealmAuthority":
        from .realm_authority import DefaultRealmAuthority
        return DefaultRealmAuthority
    elif name == "get_default_realm_authority":
        from .realm_authority import get_default_realm_authority
        return get_default_realm_authority
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "WorldState",
    "StateDelta",
    "EventType",
    "NarrativeEvent",
    "NarrativeEventStore",
    "SnapshotManager",
    "CompressedState",
    "Predicate",
    "PredicateDelta",
    "Projector",
    "PlanningContract",
    "ContextCompiler",
    "ControlledWriter",
    "SceneCompletionService",
    "ScenePlanningService",
    "WritingService",
    "ChapterTransitionService",
    "VersionedWriter",
    "VersionedWritingResult",
    "ContractSanityGuard",
    "SanityResult",
    "SanityViolation",
    "DefaultRealmAuthority",
    "get_default_realm_authority",
]