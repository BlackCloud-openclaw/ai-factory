# src/writing/audit/registry.py
"""
Phase 10.2: Stage 和 ArtifactType 注册表（实例化，可扩展）
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class StageDefinition:
    id: str
    display_name: str = ""
    description: str = ""


@dataclass(frozen=True)
class ArtifactTypeDefinition:
    id: str
    display_name: str = ""
    description: str = ""


class StageRegistry:
    def __init__(self):
        self._stages: Dict[str, StageDefinition] = {}

    def register(self, definition: StageDefinition) -> None:
        if definition.id in self._stages:
            raise ValueError(f"Stage '{definition.id}' already registered")
        self._stages[definition.id] = definition

    def get(self, stage_id: str) -> Optional[StageDefinition]:
        return self._stages.get(stage_id)

    def is_valid(self, stage_id: str) -> bool:
        return stage_id in self._stages

    def list(self) -> list[str]:
        return list(self._stages.keys())


class ArtifactTypeRegistry:
    def __init__(self):
        self._types: Dict[str, ArtifactTypeDefinition] = {}

    def register(self, definition: ArtifactTypeDefinition) -> None:
        if definition.id in self._types:
            raise ValueError(f"ArtifactType '{definition.id}' already registered")
        self._types[definition.id] = definition

    def get(self, type_id: str) -> Optional[ArtifactTypeDefinition]:
        return self._types.get(type_id)

    def is_valid(self, type_id: str) -> bool:
        return type_id in self._types

    def list(self) -> list[str]:
        return list(self._types.keys())


def create_default_stage_registry() -> StageRegistry:
    registry = StageRegistry()
    registry.register(StageDefinition("planning", "规划", "场景规划"))
    registry.register(StageDefinition("observation", "观察", "观察编译"))
    registry.register(StageDefinition("ir", "IR", "中间表示构建"))
    registry.register(StageDefinition("prompt", "Prompt", "Prompt 渲染"))
    registry.register(StageDefinition("draft", "草稿", "草稿生成"))
    registry.register(StageDefinition("coverage", "覆盖度", "覆盖度检查"))
    registry.register(StageDefinition("writer", "Writer", "Writer 执行"))  # 新增
    return registry


def create_default_artifact_type_registry() -> ArtifactTypeRegistry:
    registry = ArtifactTypeRegistry()
    registry.register(ArtifactTypeDefinition("planning", "规划结果"))
    registry.register(ArtifactTypeDefinition("observation_ir", "观察 IR"))
    registry.register(ArtifactTypeDefinition("writer_ir", "Writer IR"))
    registry.register(ArtifactTypeDefinition("prompt_bundle", "Prompt 包"))
    registry.register(ArtifactTypeDefinition("draft", "草稿"))
    registry.register(ArtifactTypeDefinition("coverage", "覆盖度报告"))
    registry.register(ArtifactTypeDefinition("writer_result", "Writer 执行结果"))
    return registry