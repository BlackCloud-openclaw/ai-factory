# src/writing/snapshot/runtime/remote/gc/existence.py
"""
B4.9/B4.10: ChunkExistenceChecker 实现
"""

import logging
from typing import final, Set, Optional

from ...chunk_ref import ChunkRef
from ..s3.client import S3Client
from ..s3.key_layout import S3KeyLayout
from ..s3.errors import S3NotFoundError
from .capability import ChunkExistenceChecker, ChunkEnumerator

logger = logging.getLogger(__name__)


@final
class S3ChunkExistenceChecker:
    """基于 S3 head_object 的 O(1) 空间检查。"""

    def __init__(self, client: S3Client, key_layout: S3KeyLayout):
        self._client = client
        self._key_layout = key_layout

    def exists(self, ref: ChunkRef) -> bool:
        key = self._key_layout.chunk_key(ref.snapshot_id, ref.chunk_id)
        try:
            self._client.head_object(key)
            return True
        except S3NotFoundError:
            return False


@final
class EnumeratorExistenceChecker:
    """
    将 ChunkEnumerator 适配为 ChunkExistenceChecker。

    生命周期警告：
        此适配器在第一次调用 `exists()` 时会完整调用 `list_all_chunks()`，
        并将结果缓存为 `Set[ChunkRef]`。之后的所有 `exists()` 调用都基于该缓存。

    因此，此适配器 **仅适用于单次 reconcile 会话**，不应在多次 reconcile 之间复用。
    """

    def __init__(self, enumerator: ChunkEnumerator):
        self._enumerator = enumerator
        self._cache: Set[ChunkRef] | None = None

    def exists(self, ref: ChunkRef) -> bool:
        if self._cache is None:
            self._cache = set(self._enumerator.list_all_chunks())
        return ref in self._cache