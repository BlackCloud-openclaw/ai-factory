"""
FailureProvider Protocol
"""

from typing import Protocol, Iterator
from .models import RawFailureRecord


class FailureProvider(Protocol):
    """失败记录提供者协议"""

    def iter_records(self) -> Iterator[RawFailureRecord]:
        """流式迭代所有记录（无状态，可多次调用）"""
        ...