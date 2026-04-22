import pytest
from src.agents.research import ResearchAgent


class TestResearchAgent:
    def test_init(self):
        agent = ResearchAgent()
        assert agent.top_k == 10

    def test_init_custom_params(self):
        agent = ResearchAgent(top_k=5, llm_api_url="http://test:8080")
        assert agent.top_k == 5
        assert agent.llm_api_url == "http://test:8080"

    def test_run_no_retriever_results(self, monkeypatch):
        """Test that agent handles empty retriever results gracefully."""
        async def mock_search(*args, **kwargs):
            return []

        monkeypatch.setattr(
            "src.knowledge.retrieval.KnowledgeRetriever.search",
            mock_search,
        )

        agent = ResearchAgent()
        result = pytest.importorskip("asyncio").run(agent.run("test query"))
        assert "summary" in result
        assert result["summary"] == "No relevant information found in the knowledge base."
        assert result["sources"] == []
