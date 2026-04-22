import pytest
from src.knowledge.reranker import Reranker


class TestReranker:
    def test_rerank_empty_results(self):
        reranker = Reranker()
        result = pytest.raises(Exception)
        # The async method needs to be run in an event loop
        import asyncio
        result = asyncio.run(reranker.rerank("query", []))
        assert result == []

    def test_rerank_single_result(self):
        reranker = Reranker()
        results = [
            {"content": "test content", "score": 0.8},
        ]
        import asyncio
        result = asyncio.run(reranker.rerank("test", results))
        assert len(result) == 1
        assert "score" in result[0]

    def test_rerank_multiple_results_sorted(self):
        reranker = Reranker()
        results = [
            {"content": "first content", "score": 0.9},
            {"content": "second content", "score": 0.5},
            {"content": "third content", "score": 0.7},
        ]
        import asyncio
        result = asyncio.run(reranker.rerank("test", results))
        # Results should be sorted by score (descending)
        assert len(result) == 3
        assert result[0]["score"] >= result[1]["score"]

    def test_rerank_with_top_k(self):
        reranker = Reranker()
        results = [
            {"content": f"content {i}", "score": float(i)}
            for i in range(10)
        ]
        import asyncio
        result = asyncio.run(reranker.rerank("test", results, top_k=3))
        assert len(result) == 3

    def test_rerank_with_metadata_filter(self):
        reranker = Reranker()
        results = [
            {"content": "test", "score": 0.9, "metadata": {"source": "doc1"}},
        ]
        import asyncio
        result = asyncio.run(reranker.rerank_with_metadata("test", results, top_k=1))
        # Should filter by threshold (default 0.5)
        assert isinstance(result, list)
