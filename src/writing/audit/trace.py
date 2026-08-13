# src/writing/audit/trace.py
"""
Phase 10.2: Execution Trace — Writer Runtime 执行的完整记录（深不可变 DAG）
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Set, Tuple, Mapping, FrozenSet
from uuid import UUID, uuid4

from .registry import StageRegistry, ArtifactTypeRegistry
from .payload_ref import PayloadRef

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class Artifact:
    """
    工件（轻量级引用，不存储实际数据）。

    Attributes:
        artifact_id: 唯一标识
        artifact_type: 工件类型（注册表 ID）
        payload_ref: 运行时对象引用
        digest: 内容摘要（用于去重和完整性）
        size_bytes: 数据大小（字节）
        metadata: 额外元数据
    """
    artifact_id: UUID = field(default_factory=uuid4)
    artifact_type: str = ""
    payload_ref: PayloadRef = field(default_factory=lambda: PayloadRef(""))
    digest: str = ""
    size_bytes: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # 深不可变：确保 metadata 是 MappingProxyType
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(
                self,
                "metadata",
                MappingProxyType(dict(self.metadata)),
            )


@dataclass(frozen=True)
class StageRecord:
    """
    阶段执行记录（不可变）。

    Attributes:
        stage: 阶段 ID（注册表）
        stage_id: 阶段唯一标识
        input_artifacts: 输入工件 ID 列表
        output_artifacts: 输出工件 ID 列表
        start_time: 开始时间（UTC）
        end_time: 结束时间（UTC）
        metrics: 指标字典（tokens, cost, 等）
    """
    stage: str
    stage_id: UUID = field(default_factory=uuid4)
    input_artifacts: Tuple[UUID, ...] = field(default_factory=tuple)
    output_artifacts: Tuple[UUID, ...] = field(default_factory=tuple)
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.input_artifacts, tuple):
            object.__setattr__(self, "input_artifacts", tuple(self.input_artifacts))
        if not isinstance(self.output_artifacts, tuple):
            object.__setattr__(self, "output_artifacts", tuple(self.output_artifacts))
        # 深不可变：metrics 必须是 MappingProxyType
        if not isinstance(self.metrics, MappingProxyType):
            object.__setattr__(
                self,
                "metrics",
                MappingProxyType(dict(self.metrics)),
            )

    @property
    def duration_ms(self) -> Optional[int]:
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return int(delta.total_seconds() * 1000)


@dataclass(frozen=True)
class ExecutionTrace:
    """
    Writer Runtime 执行的完整追踪记录（不可变，深不可变）。

    Attributes:
        execution_id: 执行唯一标识
        schema_version: Trace Schema 版本
        novel_id: 小说 ID
        volume: 卷号
        chapter: 章号
        scene_idx: 场景索引
        stages: 阶段记录列表
        artifacts: 工件字典（artifact_id -> Artifact）
        start_time: 执行开始时间
        end_time: 执行结束时间
        metadata: 额外元数据
    """
    execution_id: UUID
    schema_version: str = SCHEMA_VERSION
    novel_id: str = ""
    volume: int = 0
    chapter: int = 0
    scene_idx: int = 0
    stages: Tuple[StageRecord, ...] = field(default_factory=tuple)
    artifacts: Mapping[UUID, Artifact] = field(default_factory=dict)
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # 私有索引（深不可变，仅在 __post_init__ 构建）
    _producer: Mapping[UUID, UUID] = field(init=False, repr=False, default=None)
    _consumer: Mapping[UUID, Tuple[UUID, ...]] = field(init=False, repr=False, default=None)
    _upstream: Mapping[UUID, FrozenSet[UUID]] = field(init=False, repr=False, default=None)
    _downstream: Mapping[UUID, FrozenSet[UUID]] = field(init=False, repr=False, default=None)
    _stage_index: Mapping[UUID, StageRecord] = field(init=False, repr=False, default=None)
    _latest_by_name: Mapping[str, StageRecord] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        # 1. 确保 stages 是 tuple
        if not isinstance(self.stages, tuple):
            object.__setattr__(self, "stages", tuple(self.stages))

        # 2. 深不可变：artifacts 必须是 MappingProxyType（复制后包装）
        if not isinstance(self.artifacts, MappingProxyType):
            object.__setattr__(
                self,
                "artifacts",
                MappingProxyType(dict(self.artifacts)),
            )

        # 3. 深不可变：metadata 必须是 MappingProxyType
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(
                self,
                "metadata",
                MappingProxyType(dict(self.metadata)),
            )

        # 4. 构建索引（仅一次）
        producer = self._build_producer()
        consumer = self._build_consumer()
        upstream = self._build_upstream(producer)
        downstream = self._build_downstream(producer, consumer)
        stage_index = self._build_stage_index()
        latest_by_name = self._build_latest_by_name()

        object.__setattr__(self, "_producer", producer)
        object.__setattr__(self, "_consumer", consumer)
        object.__setattr__(self, "_upstream", upstream)
        object.__setattr__(self, "_downstream", downstream)
        object.__setattr__(self, "_stage_index", stage_index)
        object.__setattr__(self, "_latest_by_name", latest_by_name)

    # ========== 索引构建（私有，O(E)） ==========

    def _build_producer(self) -> Mapping[UUID, UUID]:
        """构建 artifact_id -> producing stage_id 索引，并检测重复 producer。"""
        producer: Dict[UUID, UUID] = {}
        for stage in self.stages:
            for out_id in stage.output_artifacts:
                if out_id in producer:
                    existing_stage = producer[out_id]
                    raise ValueError(
                        f"Artifact {out_id} is produced by multiple stages: "
                        f"{existing_stage} and {stage.stage_id}"
                    )
                producer[out_id] = stage.stage_id
        return MappingProxyType(producer)

    def _build_consumer(self) -> Mapping[UUID, Tuple[UUID, ...]]:
        """构建 artifact_id -> consuming stage_ids 索引。"""
        consumer: Dict[UUID, List[UUID]] = {}
        for stage in self.stages:
            for in_id in stage.input_artifacts:
                consumer.setdefault(in_id, []).append(stage.stage_id)
        return MappingProxyType({k: tuple(v) for k, v in consumer.items()})

    def _build_upstream(self, producer: Mapping[UUID, UUID]) -> Mapping[UUID, FrozenSet[UUID]]:
        """构建 stage_id -> upstream stage_ids 索引。"""
        upstream: Dict[UUID, Set[UUID]] = {s.stage_id: set() for s in self.stages}
        for stage in self.stages:
            for in_id in stage.input_artifacts:
                prod = producer.get(in_id)
                if prod is not None:
                    upstream[stage.stage_id].add(prod)
        return MappingProxyType({k: frozenset(v) for k, v in upstream.items()})

    def _build_downstream(
        self,
        producer: Mapping[UUID, UUID],
        consumer: Mapping[UUID, Tuple[UUID, ...]],
    ) -> Mapping[UUID, FrozenSet[UUID]]:
        """
        构建 stage_id -> downstream stage_ids 索引（O(E)）。
        基于 producer + consumer 一次性构建。
        """
        downstream: Dict[UUID, Set[UUID]] = {s.stage_id: set() for s in self.stages}
        for artifact_id, consumer_stages in consumer.items():
            prod_stage_id = producer.get(artifact_id)
            if prod_stage_id is not None:
                for consumer_stage_id in consumer_stages:
                    downstream[prod_stage_id].add(consumer_stage_id)
        return MappingProxyType({k: frozenset(v) for k, v in downstream.items()})

    def _build_stage_index(self) -> Mapping[UUID, StageRecord]:
        """构建 stage_id -> StageRecord 索引，并检测重复 stage_id。"""
        index: Dict[UUID, StageRecord] = {}
        for stage in self.stages:
            if stage.stage_id in index:
                raise ValueError(
                    f"Duplicate stage_id: {stage.stage_id}. "
                    "Each stage must have a unique ID."
                )
            index[stage.stage_id] = stage
        return MappingProxyType(index)

    def _build_latest_by_name(self) -> Mapping[str, StageRecord]:
        """构建 stage_name -> 最新 StageRecord 索引。"""
        latest: Dict[str, StageRecord] = {}
        for s in self.stages:
            latest[s.stage] = s
        return MappingProxyType(latest)

    # ========== 基本查询 ==========

    @property
    def total_duration_ms(self) -> Optional[int]:
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return int(delta.total_seconds() * 1000)

    def get_stage(self, stage_name: str) -> Optional[StageRecord]:
        """返回指定名称的最后一个阶段记录（O(1)）。"""
        return self._latest_by_name.get(stage_name)

    def find_stages(self, stage_name: str) -> List[StageRecord]:
        """返回指定名称的所有阶段记录（按执行顺序）。"""
        return [s for s in self.stages if s.stage == stage_name]

    def get_stage_by_uuid(self, stage_uuid: UUID) -> Optional[StageRecord]:
        """O(1) 通过 UUID 获取阶段记录。"""
        return self._stage_index.get(stage_uuid)

    def get_artifact(self, artifact_id: UUID) -> Optional[Artifact]:
        return self.artifacts.get(artifact_id)

    def get_stage_artifacts(self, stage_name: str, role: str = "output") -> List[Artifact]:
        record = self.get_stage(stage_name)
        if record is None:
            return []
        ids = record.output_artifacts if role == "output" else record.input_artifacts
        return [self.artifacts.get(aid) for aid in ids if aid in self.artifacts]

    # ========== 图查询 API ==========

    def parent_ids(self, artifact_id: UUID) -> Tuple[UUID, ...]:
        """返回直接上游 artifacts 的 ID（O(1)）。"""
        prod_stage_id = self._producer.get(artifact_id)
        if prod_stage_id is None:
            return ()
        stage = self.get_stage_by_uuid(prod_stage_id)
        if stage is None:
            return ()
        return stage.input_artifacts

    def child_ids(self, artifact_id: UUID) -> Tuple[UUID, ...]:
        """返回直接下游 artifacts 的 ID（O(1)）。"""
        consumer_stage_ids = self._consumer.get(artifact_id, ())
        if not consumer_stage_ids:
            return ()
        result: List[UUID] = []
        for stage_uuid in consumer_stage_ids:
            stage = self.get_stage_by_uuid(stage_uuid)
            if stage is not None:
                result.extend(stage.output_artifacts)
        return tuple(result)

    def parents(self, artifact_id: UUID) -> List[Artifact]:
        """返回直接上游 artifacts。"""
        return [self.artifacts.get(aid) for aid in self.parent_ids(artifact_id) if aid in self.artifacts]

    def children(self, artifact_id: UUID) -> List[Artifact]:
        """返回直接下游 artifacts。"""
        return [self.artifacts.get(aid) for aid in self.child_ids(artifact_id) if aid in self.artifacts]

    def upstream(self, stage_uuid: UUID) -> List[StageRecord]:
        """获取指定阶段的所有上游阶段（沿输入追溯）。"""
        visited: Set[UUID] = set()
        result: List[StageRecord] = []
        self._upstream_recursive(stage_uuid, visited, result)
        return result

    def _upstream_recursive(self, stage_uuid: UUID, visited: Set[UUID], result: List[StageRecord]):
        if stage_uuid in visited:
            return
        visited.add(stage_uuid)
        for up_uuid in self._upstream.get(stage_uuid, frozenset()):
            stage = self.get_stage_by_uuid(up_uuid)
            if stage is not None:
                result.append(stage)
                self._upstream_recursive(up_uuid, visited, result)

    def downstream(self, stage_uuid: UUID) -> List[StageRecord]:
        """获取指定阶段的所有下游阶段（沿输出追溯）。"""
        visited: Set[UUID] = set()
        result: List[StageRecord] = []
        self._downstream_recursive(stage_uuid, visited, result)
        return result

    def _downstream_recursive(self, stage_uuid: UUID, visited: Set[UUID], result: List[StageRecord]):
        if stage_uuid in visited:
            return
        visited.add(stage_uuid)
        for down_uuid in self._downstream.get(stage_uuid, frozenset()):
            stage = self.get_stage_by_uuid(down_uuid)
            if stage is not None:
                result.append(stage)
                self._downstream_recursive(down_uuid, visited, result)

    # ========== 便捷方法 ==========

    def upstream_by_name(self, stage_name: str) -> List[StageRecord]:
        stage = self.get_stage(stage_name)
        if stage is None:
            return []
        return self.upstream(stage.stage_id)

    def downstream_by_name(self, stage_name: str) -> List[StageRecord]:
        stage = self.get_stage(stage_name)
        if stage is None:
            return []
        return self.downstream(stage.stage_id)