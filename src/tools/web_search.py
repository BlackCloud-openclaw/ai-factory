"""Simple web search tool using DuckDuckGo (ddgs library)."""

import asyncio
from typing import List, Dict, Any
from ddgs import DDGS

async def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Perform a web search using DuckDuckGo.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default 5)

    Returns:
        List of dicts with keys: title, body, href, source
    """
    # DDGS is sync; run in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: list(DDGS().text(query, max_results=max_results))
    )
    # Standardize output
    return [
        {
            "title": r.get("title", ""),
            "body": r.get("body", ""),
            "href": r.get("href", ""),
            "source": "duckduckgo"
        }
        for r in results
    ]
