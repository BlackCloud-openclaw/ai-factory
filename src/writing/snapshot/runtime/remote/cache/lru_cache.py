# src/writing/snapshot/runtime/remote/cache/lru_cache.py
"""
B4.2: LRUCache — 线程安全的 LRU 缓存（基于 OrderedDict）
"""

from collections import OrderedDict
from threading import Lock
from typing import Generic, Optional, TypeVar, Tuple

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """
    线程安全的 LRU 缓存。

    使用 OrderedDict 实现 O(1) 的 get/put 操作。
    """

    def __init__(self, maxsize: int = 128):
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")

        self._maxsize = maxsize
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._lock = Lock()

        # 指标
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def lookup(self, key: K) -> Tuple[bool, Optional[V]]:
        """
        查找缓存项，并返回 (是否存在, 值)。
        命中时更新访问顺序。
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return True, self._cache[key]
            self._misses += 1
            return False, None

    def put(self, key: K, value: V) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
                return

            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
                self._evictions += 1

            self._cache[key] = value

    def invalidate(self, key: K) -> None:
        with self._lock:
            if key in self._cache:
                del self._cache[key]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def contains(self, key: K) -> bool:
        with self._lock:
            return key in self._cache

    def keys(self) -> list[K]:
        with self._lock:
            return list(self._cache.keys())

    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def maxsize(self) -> int:
        return self._maxsize

    def hits(self) -> int:
        with self._lock:
            return self._hits

    def misses(self) -> int:
        with self._lock:
            return self._misses

    def evictions(self) -> int:
        with self._lock:
            return self._evictions

    def reset_metrics(self) -> None:
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0