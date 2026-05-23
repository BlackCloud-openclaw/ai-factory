import pytest
from src.writing.causality.budget import ConsistencyBudget
from src.db import init_db_pool, close_db_pool

@pytest.mark.asyncio
async def test_budget_consume():
    await init_db_pool()
    try:
        budget = ConsistencyBudget("test_novel", 1, 1)
        await budget.reset()
        result = await budget.consume("warning")
        assert result is True
        assert budget.remaining_warnings == 2
    finally:
        await close_db_pool()