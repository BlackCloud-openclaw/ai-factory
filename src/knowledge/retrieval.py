"""Vector retrieval module using PostgreSQL + pgvector with HNSW indexing."""

import json
from typing import Any, Optional

import psycopg2
import psycopg2.pool
import psycopg2.extras
import numpy as np

from src.config import config
from src.common.logging import setup_logging
from src.common.retry import retry_with_backoff

logger = setup_logging("knowledge.retrieval")


class KnowledgeRetriever:
    """Retrieve relevant document chunks from pgvector store."""

    def __init__(
        self,
        dsn: Optional[str] = None,
        embedding_dim: int = config.embedding_dim,
        top_k: int = config.rag_k,
    ):
        self.dsn = dsn or config.postgres_dsn
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self._connection_pool: Optional[psycopg2.pool.SimpleConnectionPool] = None

    def connect(self):
        """Initialize connection pool."""
        if self._connection_pool is None:
            self._connection_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=config.postgres_pool_size,
                dsn=self.dsn,
            )
            logger.info("Connected to PostgreSQL with pgvector")

    def close(self):
        """Close all connections in the pool."""
        if self._connection_pool:
            self._connection_pool.closeall()
            self._connection_pool = None
            logger.info("Closed PostgreSQL connection pool")

    def _get_conn(self):
        """Get a connection from the pool."""
        if self._connection_pool is None:
            self.connect()
        return self._connection_pool.getconn()

    def _put_conn(self, conn):
        """Return a connection to the pool."""
        if self._connection_pool:
            self._connection_pool.putconn(conn)

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def search(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Search for relevant chunks using vector similarity.

        Note: Embedding generation should be done externally.
        This method expects pre-computed query embeddings.
        Use search_with_embedding() for vector search.
        """
        # For now, fall back to keyword search
        return await self._keyword_search(query, k or self.top_k, filter_metadata)

    async def search_with_embedding(
        self,
        query_embedding: list[float],
        k: Optional[int] = None,
        filter_metadata: Optional[dict[str, Any]] = None,
        threshold: float = config.rerank_threshold,
    ) -> list[dict[str, Any]]:
        """Search using pre-computed query embedding with HNSW vector cosine similarity."""
        conn = self._get_conn()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Convert embedding to PG vector array format
            embedding_str = f"[{','.join(f'{x:.8f}' for x in query_embedding)}]"

            query = """
                SELECT
                    id AS chunk_id,
                    content,
                    document_id,
                    metadata,
                    1 - (embedding <=> %(embedding)s::vector) AS score
                FROM chunks
                WHERE embedding IS NOT NULL
            """

            params = {"embedding": embedding_str}

            if filter_metadata:
                for key, value in filter_metadata.items():
                    query += f" AND metadata->>'{key}' = %s"
                    params[key] = str(value)

            query += f" ORDER BY score DESC LIMIT %s"
            params["limit"] = k or self.top_k

            cur.execute(query, params)
            results = cur.fetchall()

            # Filter by threshold
            filtered = [
                dict(r)
                for r in results
                if float(r["score"]) >= threshold
            ]

            # Convert numpy types to native Python types
            for r in filtered:
                r["score"] = float(r["score"])
                if isinstance(r.get("metadata"), str):
                    r["metadata"] = json.loads(r["metadata"])

            logger.info(f"Vector search returned {len(filtered)} results")
            return filtered

        finally:
            self._put_conn(conn)

    async def _keyword_search(
        self,
        query: str,
        k: int = 10,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Fallback keyword search using tsvector."""
        conn = self._get_conn()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            query_sql = """
                SELECT
                    id AS chunk_id,
                    content,
                    document_id,
                    metadata,
                    ts_rank(to_tsvector(content), plainto_tsquery(%s)) AS relevance
                FROM chunks
                WHERE to_tsvector(content) @@ plainto_tsquery(%s)
            """

            params = [query, query]

            if filter_metadata:
                for key, value in filter_metadata.items():
                    query_sql += f" AND metadata->>'{key}' = %s"
                    params.append(str(value))

            query_sql += f" ORDER BY relevance DESC LIMIT %s"
            params.append(k)

            cur.execute(query_sql, params)
            results = cur.fetchall()

            for r in results:
                r["score"] = float(r["relevance"])
                if isinstance(r.get("metadata"), str):
                    r["metadata"] = json.loads(r["metadata"])

            logger.info(f"Keyword search returned {len(results)} results")
            return [
                {
                    "chunk_id": r["chunk_id"],
                    "content": r["content"],
                    "document_id": r["document_id"],
                    "score": r["score"],
                    "metadata": r.get("metadata", {}),
                }
                for r in results
            ]

        finally:
            self._put_conn(conn)

    async def store_chunks(
        self,
        chunks: list[dict[str, Any]],
        document_id: str,
        embeddings: Optional[list[list[float]]] = None,
    ) -> int:
        """Store document chunks in the database."""
        conn = self._get_conn()
        inserted = 0
        try:
            cur = conn.cursor()

            for i, chunk in enumerate(chunks):
                embedding = None
                if embeddings and i < len(embeddings):
                    emb = embeddings[i]
                    embedding = f"[{','.join(f'{x:.8f}' for x in emb)}]"

                metadata = json.dumps(chunk.get("metadata", {}))

                cur.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s::vector, %s)
                    ON CONFLICT (document_id, chunk_index)
                    DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata
                    """,
                    (
                        document_id,
                        chunk.get("chunk_index", i),
                        chunk["content"],
                        embedding,
                        metadata,
                    ),
                )
                inserted += 1

            conn.commit()
            logger.info(f"Stored {inserted} chunks for document {document_id}")
            return inserted

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to store chunks: {e}")
            raise
        finally:
            self._put_conn(conn)

    async def store_document(
        self,
        doc_id: str,
        title: str,
        source_path: str,
        file_type: str,
        content: str,
    ) -> None:
        """Store a document record."""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO documents (id, title, source_path, file_type, content)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (doc_id, title, source_path, file_type, content),
            )
            conn.commit()
        finally:
            self._put_conn(conn)
