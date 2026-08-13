# tests/unit/snapshot/runtime/remote/test_cache_metrics.py

import threading
from typing import Optional  # 添加导入

import pytest

from src.writing.snapshot.runtime.remote.cache import LRUCache, CacheMetrics


class TestLRUCache:
    def test_basic_get_put(self):
        cache = LRUCache[int, str](maxsize=3)
        cache.put(1, "a")
        cache.put(2, "b")

        found, val = cache.lookup(1)
        assert found is True and val == "a"
        found, val = cache.lookup(2)
        assert found is True and val == "b"
        found, val = cache.lookup(3)
        assert found is False and val is None

        assert cache.hits() == 2
        assert cache.misses() == 1

    def test_eviction_and_metrics(self):
        cache = LRUCache[int, str](maxsize=2)
        cache.put(1, "a")
        cache.put(2, "b")
        cache.put(3, "c")

        assert cache.evictions() == 1
        assert cache.size() == 2
        assert cache.contains(1) is False
        assert cache.contains(2) is True
        assert cache.contains(3) is True

        cache.lookup(2)
        cache.put(4, "d")
        assert cache.contains(3) is False
        assert cache.evictions() == 2

    def test_none_value(self):
        cache = LRUCache[int, Optional[str]](maxsize=2)
        cache.put(1, None)
        found, val = cache.lookup(1)
        assert found is True and val is None

        found, val = cache.lookup(2)
        assert found is False and val is None

    def test_concurrent_access(self):
        cache = LRUCache[int, int](maxsize=100)
        errors = []

        def worker():
            for i in range(100):
                cache.put(i, i)
                found, val = cache.lookup(i)
                if not found or val != i:
                    errors.append(f"Miss match for {i}")

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert cache.size() <= 100