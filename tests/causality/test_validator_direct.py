import pytest
from src.writing.causality.validator import CausalityValidator


@pytest.mark.asyncio
async def test_causality_validator():
    validator = CausalityValidator()
    # 简单测试：空事件应该通过
    result = validator.validate({}, {})
    assert result["passed"] is True