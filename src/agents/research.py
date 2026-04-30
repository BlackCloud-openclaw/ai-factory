# src/agents/research.py
import asyncio
import json
import time
from typing import Any, Optional, List, Dict

from src.knowledge.retrieval import KnowledgeRetriever
from src.knowledge.reranker import Reranker
from src.tools.web_search import web_search
from src.config import config
from src.common.logging import setup_logging
from src.common.retry import retry_with_backoff
from src.agents.base import BaseAgent
from src.orchestrator.state import AgentState
from src.config import config

logger = setup_logging("agents.research")

class ResearchAgent(BaseAgent):
    def __init__(
        self,
        retriever: Optional[KnowledgeRetriever] = None,
        reranker: Optional[Reranker] = None,
        llm_api_url: str = config.llm_api_url,
        llm_model: str = config.llm_model_name,
        top_k: int = config.rag_k,
        enable_web_search: bool = True,
    ):
        self.retriever = retriever or KnowledgeRetriever()
        # 禁用 Reranker，避免网络依赖
        self.reranker = None  # 原为 reranker or Reranker()
        self.llm_api_url = llm_api_url
        self.llm_model = llm_model
        self.top_k = top_k
        self.enable_web_search = enable_web_search
        self.system_prompt = (
            "You are a knowledgeable research assistant. "
            "Use the provided context to answer the user's question accurately. "
            "If the context doesn't contain enough information, say so clearly. "
            "Cite your sources when possible."
        )

    async def run(self, state: AgentState) -> Dict[str, Any]:
        agent_name = "ResearchAgent"
        state.step_count += 1
        step = state.step_count
        logger.info(f"Starting {agent_name}, step={step}")
        start_time = time.time()
        query = state.user_input or state.original_request
        logger.info(f"ResearchAgent running for query: {query[:150]}")

        try:
            local_chunks = await self.retriever.search(query, k=self.top_k)
            logger.info(f"Local KB returned {len(local_chunks)} chunks")

            web_results = []
            if self.enable_web_search and len(local_chunks) < 2:
                try:
                    web_results = await web_search(query, max_results=3)
                    logger.info(f"Web search returned {len(web_results)} results")
                except Exception as e:
                    logger.warning(f"Web search failed: {e}")

            all_results = []
            for c in local_chunks:
                all_results.append({
                    "content": c.get("content", ""),
                    "score": c.get("score", 0.0),
                    "source": c.get("metadata", {}).get("source_path", "local_kb"),
                    "type": "local"
                })
            for w in web_results:
                all_results.append({
                    "content": f"Title: {w['title']}\nBody: {w['body']}",
                    "score": 0.5,
                    "source": w.get("href", w.get("source", "web")),
                    "type": "web"
                })

            if not all_results:
                summary = "No relevant information found."
                research_results = []
                sources = []
            else:
                # Reranker 已禁用，跳过重排序
                if self.reranker and len(all_results) > 1:
                    try:
                        all_results = await asyncio.to_thread(
                            self.reranker.rerank, query, all_results, 5
                        )
                        logger.info(f"Reranked results, kept top {len(all_results)}")
                    except Exception as e:
                        logger.warning(f"Reranker failed: {e}")

                context = self._build_context(all_results)
                summary = await self._summarize(query, context) or "Summary generation failed, but search results are available."
                research_results = []
                sources = []
                for r in all_results[:5]:
                    research_results.append({
                        "summary": r["content"][:500],
                        "source": r["source"],
                        "score": r.get("score", 0.0),
                        "type": r.get("type", "unknown")
                    })
                    sources.append({"source_path": r["source"], "type": r.get("type")})

            return {
                "research_results": research_results,
                "sources": sources,
                "final_answer": summary,
            }
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{agent_name} failed, step={step}, error={e}", exc_info=True)
            return {
                "research_results": [],
                "sources": [],
                "final_answer": f"Research encountered an error: {str(e)}",
            }

    def _build_context(self, results: List[Dict[str, Any]]) -> str:
        parts = []
        for i, r in enumerate(results, 1):
            src = r.get("source", f"source_{i}")
            content = r.get("content", "")
            parts.append(f"[{i}] Source: {src}\n{content}")
        return "\n\n---\n\n".join(parts)

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    async def _summarize(self, query: str, context: str) -> str:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key="not-needed", base_url=self.llm_api_url)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}\n\nPlease provide a concise and accurate answer based on the context."}
        ]
        try:
            response = await client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.2,
                max_tokens=config.llm_max_tokens,
                timeout=config.llm_timeout_research,
            )
            return response.choices[0].message.content or "No summary generated."
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
            return f"Unable to generate summary due to LLM error. Raw context length: {len(context)} characters."