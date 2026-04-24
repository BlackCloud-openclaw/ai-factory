import json
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage

from src.knowledge.retrieval import KnowledgeRetriever
from src.config import config
from src.common.logging import setup_logging
from src.common.retry import retry_with_backoff
from src.orchestrator.state import AgentState

logger = setup_logging("agents.research")


class ResearchAgent:
    """Agent responsible for knowledge retrieval and LLM-based summarization."""

    def __init__(
        self,
        retriever: Optional[KnowledgeRetriever] = None,
        llm_api_url: str = config.llm_api_url,
        llm_model: str = config.llm_model_name,
        top_k: int = config.rag_k,
    ):
        self.retriever = retriever or KnowledgeRetriever()
        self.llm_api_url = llm_api_url
        self.llm_model = llm_model
        self.top_k = top_k
        self.system_prompt = (
            "You are a knowledgeable research assistant. "
            "Use the provided context to answer the user's question accurately. "
            "If the context doesn't contain enough information, say so clearly. "
            "Cite your sources when possible."
        )

    async def run(self, state: AgentState) -> Dict[str, Any]:
        """Execute research based on the current agent state."""
        query = state.user_input or state.original_request
        if not query:
            logger.warning("ResearchAgent: no query found in state")
            return {"research_results": [], "sources": []}

        logger.info(f"ResearchAgent running for query: {query[:150]}")

        # Step 1: Retrieve relevant chunks from knowledge base
        chunks = await self.retriever.search(query, k=self.top_k)

        if not chunks:
            logger.warning(f"No knowledge base results for query: {query}")
            return {"research_results": [], "sources": []}

        # Step 2: Build context from retrieved chunks
        context = self._build_context(chunks)

        # Step 3: Generate summary via LLM
        summary = await self._summarize(query, context)

        sources = [
            {
                "document_id": c.get("document_id", ""),
                "score": c.get("score", 0.0),
                "source_path": c.get("metadata", {}).get("source_path", "unknown"),
            }
            for c in chunks
        ]

        research_results = [
            {
                "summary": summary,
                "sources": sources,
                "chunks": [
                    {
                        "content": c.get("content", "")[:500],
                        "score": c.get("score", 0.0),
                    }
                    for c in chunks
                ],
            }
        ]

        result = {
            "research_results": research_results,
            "sources": sources,
        }

        logger.info(f"ResearchAgent completed. Found {len(chunks)} relevant chunks.")
        return result

    def _build_context(self, chunks: list[dict[str, Any]]) -> str:
        """Build a context string from retrieved chunks."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "")
            source = chunk.get("metadata", {}).get("source_path", f"document_{i}")
            parts.append(f"[Source: {source}]\n{content}")
        return "\n\n---\n\n".join(parts)

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def _summarize(self, query: str, context: str) -> str:
        """Call LLM to generate a summary from retrieved context."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key="not-needed",  # llama.cpp doesn't require an API key
            base_url=self.llm_api_url,
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"Context:\n{context}\n\n"
                    f"Please provide a concise and accurate answer based on the context."
                ),
            },
        ]

        response = await client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.2,
            max_tokens=config.llm_max_tokens,
        )

        return response.choices[0].message.content or "No summary generated."
