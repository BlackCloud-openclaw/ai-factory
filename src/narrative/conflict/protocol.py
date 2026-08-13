# src/narrative/conflict/protocol.py

from typing import Protocol, Tuple, runtime_checkable
from src.narrative.intent.model import NarrativeIntent
from src.narrative.intent.conflict import Conflict
from src.narrative.conflict.model import ConflictResolution


@runtime_checkable
class ConflictResolver(Protocol):
    def resolve(
        self,
        conflicts: Tuple[Conflict, ...],
        intents: Tuple[NarrativeIntent, ...],
    ) -> Tuple[ConflictResolution, ...]:
        """
        输入已检测到的冲突和所有意图，返回决议列表。
        决议数量必须与 conflicts 数量一致（一一对应）。
        """
        ...