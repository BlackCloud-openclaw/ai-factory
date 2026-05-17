"""Vector retrieval module using PostgreSQL + pgvector with asyncpg."""

import json
from typing import Any, Optional, List, Dict

from src.config import config
from src.common.logging import setup_logging
from src.common.retry import retry_with_backoff
from src.db import get_db_pool

logger = setup_logging("knowledge.retrieval")


class KnowledgeRetriever:
    """Retrieve relevant document chunks from pgvector store using asyncpg."""

    def __init__(
        self,
        embedding_dim: int = config.embedding_dim,
        top_k: int = config.rag_k,
    ):
        self.embedding_dim = embedding_dim
        self.top_k = top_k

    async def _get_pool(self):
        """Get the global asyncpg pool."""
        pool = get_db_pool()
        if pool is None:
            raise RuntimeError("Database pool not initialized")
        return pool

    async def _keyword_search(
        self,
        query: str,
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Fallback keyword search using tsvector."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Build query with optional metadata filter
            sql = """
                SELECT
                    id AS chunk_id,
                    content,
                    document_id,
                    metadata,
                    ts_rank(to_tsvector('simple', content), plainto_tsquery('simple', $1)) AS relevance
                FROM chunks
                WHERE to_tsvector('simple', content) @@ plainto_tsquery('simple', $1)
            """
            params = [query]
            if filter_metadata:
                for key, value in filter_metadata.items():
                    # jsonb field access
                    sql += f" AND metadata->>${len(params)+1} = ${len(params)+2}"
                    params.append(key)
                    params.append(str(value))
            sql += f" ORDER BY relevance DESC LIMIT ${len(params)+1}"
            params.append(k)
            rows = await conn.fetch(sql, *params)
            results = []
            for row in rows:
                metadata = row["metadata"]
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                results.append({
                    "chunk_id": row["chunk_id"],
                    "content": row["content"],
                    "document_id": row["document_id"],
                    "score": float(row["relevance"]),
                    "metadata": metadata or {},
                })
            logger.info(f"Keyword search returned {len(results)} results")
            return results

    async def _search_with_embedding(
        self,
        query_embedding: List[float],
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        threshold: float = config.rerank_threshold,
    ) -> List[Dict[str, Any]]:
        """Search using pre-computed query embedding with HNSW vector cosine similarity."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Convert embedding to pgvector string format
            embedding_str = "[" + ",".join(f"{x:.8f}" for x in query_embedding) + "]"
            sql = """
                SELECT
                    id AS chunk_id,
                    content,
                    document_id,
                    metadata,
                    1 - (embedding <=> $1::vector) AS score
                FROM chunks
                WHERE embedding IS NOT NULL
            """
            params = [embedding_str]
            if filter_metadata:
                for key, value in filter_metadata.items():
                    sql += f" AND metadata->>${len(params)+1} = ${len(params)+2}"
                    params.append(key)
                    params.append(str(value))
            limit = k or self.top_k
            sql += f" ORDER BY score DESC LIMIT ${len(params)+1}"
            params.append(limit)
            rows = await conn.fetch(sql, *params)
            results = []
            for row in rows:
                score = float(row["score"])
                if score >= threshold:
                    metadata = row["metadata"]
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    results.append({
                        "chunk_id": row["chunk_id"],
                        "content": row["content"],
                        "document_id": row["document_id"],
                        "score": score,
                        "metadata": metadata or {},
                    })
            logger.info(f"Vector search returned {len(results)} results")
            return results

    async def _store_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_id: str,
        embeddings: Optional[List[List[float]]] = None,
    ) -> int:
        """Store document chunks in the database."""
        pool = await self._get_pool()
        inserted = 0
        async with pool.acquire() as conn:
            async with conn.transaction():
                for i, chunk in enumerate(chunks):
                    embedding = None
                    if embeddings and i < len(embeddings):
                        emb = embeddings[i]
                        embedding = "[" + ",".join(f"{x:.8f}" for x in emb) + "]"
                    metadata = json.dumps(chunk.get("metadata", {}))
                    await conn.execute(
                        """
                        INSERT INTO chunks (document_id, chunk_index, content, embedding, metadata)
                        VALUES ($1, $2, $3, $4::vector, $5)
                        ON CONFLICT (document_id, chunk_index)
                        DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata
                        """,
                        document_id,
                        chunk.get("chunk_index", i),
                        chunk["content"],
                        embedding,
                        metadata,
                    )
                    inserted += 1
                logger.info(f"Stored {inserted} chunks for document {document_id}")
                return inserted

    async def _store_document(
        self,
        doc_id: str,
        title: str,
        source_path: str,
        file_type: str,
        content: str,
    ) -> None:
        """Store a document record."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO documents (id, title, source_path, file_type, content)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (id) DO NOTHING
                """,
                doc_id, title, source_path, file_type, content,
            )

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def search(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for relevant chunks using vector similarity (fallback to keyword)."""
        # For now, fall back to keyword search
        return await self._keyword_search(query, k or self.top_k, filter_metadata)

    async def search_with_embedding(
        self,
        query_embedding: List[float],
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        threshold: float = config.rerank_threshold,
    ) -> List[Dict[str, Any]]:
        """Search using pre-computed query embedding with HNSW vector cosine similarity."""
        return await self._search_with_embedding(query_embedding, k, filter_metadata, threshold)

    async def store_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_id: str,
        embeddings: Optional[List[List[float]]] = None,
    ) -> int:
        """Store document chunks in the database."""
        return await self._store_chunks(chunks, document_id, embeddings)

    async def store_document(
        self,
        doc_id: str,
        title: str,
        source_path: str,
        file_type: str,
        content: str,
    ) -> None:
        """Store a document record."""
        await self._store_document(doc_id, title, source_path, file_type, content)