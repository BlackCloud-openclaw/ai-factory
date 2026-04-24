"""Reranking module using cross-encoder reranker."""

import json
from typing import Any, Optional

from src.config import config
from src.common.logging import setup_logging

logger = setup_logging("knowledge.reranker")


class Reranker:
    """Rerank retrieved chunks using a cross-encoder reranker."""

    def __init__(
        self,
        model_name: str = config.reranker_model,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy-load the reranker model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.model_name, device=self.device
                )
                logger.info(f"Loaded reranker model: {self.model_name}")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Reranker will use score-based fallback."
                )
                self._model = "not_available"

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank a list of search results by relevance to the query.

        Args:
            query: The search query
            results: List of search results with 'content' and 'score' fields
            top_k: Number of results to return

        Returns:
            Reranked list of results with updated scores
        """
        if not results:
            return []

        top_k = top_k or len(results)

        if self._model == "not_available":
            # Fallback: sort by existing score
            sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
            return sorted_results[:top_k]

        self._load_model()

        if self._model == "not_available":
            sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
            return sorted_results[:top_k]

        # Prepare pairs for reranking
        pairs = [(query, r.get("content", "")) for r in results]

        try:
            # CrossEncoder returns higher scores for more relevant pairs
            scores = self._model.predict(pairs)

            # Attach scores to results
            for i, result in enumerate(results):
                result["rerank_score"] = float(scores[i])
                # Combine original score and rerank score
                original_score = result.get("score", 0)
                result["score"] = 0.5 * original_score + 0.5 * scores[i]

            # Sort by combined score
            sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
            logger.info(f"Reranked {len(results)} results, returning top {top_k}")
            return sorted_results[:top_k]

        except Exception as e:
            logger.error(f"Reranking failed: {e}. Falling back to score-based sort.")
            sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
            return sorted_results[:top_k]

    def rerank_with_metadata(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Rerank results and filter out results below threshold."""
        reranked = self.rerank(query, results, top_k)

        threshold = config.rerank_threshold
        filtered = [
            r for r in reranked if r.get("score", 0) >= threshold
        ]

        logger.info(
            f"Reranking with threshold {threshold}: "
            f"{len(reranked)} -> {len(filtered)} results"
        )
        return filtered
