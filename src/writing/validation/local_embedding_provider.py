# src/writing/validation/local_embedding_provider.py
"""
Local HTTP EmbeddingProvider — sync wrapper for embedding API.

Phase 15.8-fix v2:
- 加 similarity() / batch_similarity()，满足 EmbeddingProvider 协议
- 加 MAX_CHARS 截断，防 token 超限
"""

import logging
import math
from typing import List

import requests

logger = logging.getLogger(__name__)


class LocalHttpEmbeddingProvider:
    """同步 HTTP 调用本地 embedding 服务。"""

    MAX_CHARS = 400   # embedding 服务 batch_size=512 的安全上限

    def __init__(
        self,
        endpoint: str,
        dim: int = 512,
        timeout: float = 10.0,
    ):
        self.endpoint = endpoint
        self.dim = dim
        self.timeout = timeout

    def embed(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.dim
        safe_text = text[: self.MAX_CHARS]
        try:
            resp = requests.post(
                self.endpoint,
                json={"input": safe_text},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                logger.warning(
                    "[LocalEmbedding] HTTP %s: %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return [0.0] * self.dim
            data = resp.json()
            if isinstance(data, dict) and "data" in data:
                return data["data"][0]["embedding"]
            if isinstance(data, list) and data and isinstance(data[0], list):
                return data[0]
        except Exception as e:
            logger.warning("[LocalEmbedding] request failed: %s", e)
        return [0.0] * self.dim

    # ================================================================
    # Phase 15.8-fix v2: 满足 EmbeddingProvider 协议
    # ================================================================

    def similarity(self, a: str, b: str) -> float:
        """计算两段文本的余弦相似度。EmbeddingMatcher 依赖此方法。"""
        if not a or not b:
            return 0.0
        va = self.embed(a)
        vb = self.embed(b)
        return self._cosine(va, vb)

    def batch_similarity(self, a_list: List[str], b_list: List[str]) -> List[float]:
        """批量相似度（可选实现）。"""
        if len(a_list) != len(b_list):
            return [0.0] * max(len(a_list), len(b_list))
        return [self.similarity(a, b) for a, b in zip(a_list, b_list)]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)