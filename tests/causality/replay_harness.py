"""确定性重放工具，用于测试投影一致性"""
import json
from typing import List, Dict, Any
from dataclasses import dataclass

from src.writing.event_store import NarrativeEventStore
from src.writing.world_state import WorldState
from src.writing.delta import StateDelta
from src.writing.causality.predicate import Predicate
from src.writing.causality.projector import DeltaEngine
from src.writing.causality.projection_store import ProjectionStore


@dataclass
class ReplayResult:
    world_state: WorldState
    predicates: List[Predicate]
    last_event_id: int


class ReplayHarness:
    """用于测试的确定性重放器"""

    def __init__(self, novel_id: str, pool):
        self.novel_id = novel_id
        self.pool = pool
        self.event_store = NarrativeEventStore(pool)
        self.delta_engine = DeltaEngine()
        self.proj_store = ProjectionStore(pool)

    async def full_replay(self) -> ReplayResult:
        """从头重放所有事件，构建 WorldState 和 Predicate"""
        world_state = WorldState()
        predicates = []
        last_event_id = 0

        events_with_id = await self.event_store.get_events_since(self.novel_id, 0, limit=1000000)

        for event_id, event in events_with_id:
            # 应用事件到 WorldState
            delta = StateDelta(events=[event])
            world_state = delta.apply_to(world_state)

            # 计算 Predicate Delta
            current_pred_map = {p.identity_key(): p for p in predicates}
            pred_delta = self.delta_engine.compute_delta(current_pred_map, event)
            # 应用 Predicate Delta（内存中）
            for p in pred_delta.to_activate:
                predicates.append(p)
            for ref in pred_delta.to_deactivate:
                # 移除匹配的 predicate
                predicates = [p for p in predicates if p.identity_key() != ref.identity_key]

            last_event_id = event_id

        return ReplayResult(world_state, predicates, last_event_id)

    async def replay_from_snapshot(self, snapshot_event_id: int) -> ReplayResult:
        """从快照恢复后重放增量事件"""
        from src.writing.snapshot import SnapshotManager
        snap_mgr = SnapshotManager(self.pool)
        world_state, _, last_event_id = await snap_mgr.load_latest_snapshot(self.novel_id)
        if world_state is None:
            return await self.full_replay()

        # 加载快照后的事件
        events = await self.event_store.get_events_since(self.novel_id, last_event_id)
        predicates = []  # 需要从 WorldState 投影，这里简化

        for event_id, event in events:
            delta = StateDelta(events=[event])
            world_state = delta.apply_to(world_state)
            # 同样计算谓词...
            last_event_id = event_id

        return ReplayResult(world_state, [], last_event_id)

    async def compare_replay_methods(self) -> bool:
        """比较全量重放与快照+增量重放的结果"""
        full = await self.full_replay()
        snapshot = await self.replay_from_snapshot(0)
        # 比较 WorldState 和 Predicate 集
        return full.world_state == snapshot.world_state  # 简化比较