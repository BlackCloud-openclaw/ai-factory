# src/narrative/conflict/factory.py

from .strategies import PriorityResolver, BalanceResolver, SynthesisResolver
from .composite import CompositeResolver
from .selector import StrategySelector
from .protocol import ConflictResolver  # 修正：从 protocol 导入


def create_resolver(strategy: str = "priority", selector: StrategySelector = None) -> ConflictResolver:
    if strategy == "priority":
        return PriorityResolver()
    elif strategy == "balance":
        return BalanceResolver()
    elif strategy == "synthesis":
        return SynthesisResolver()
    elif strategy == "composite":
        return CompositeResolver(selector)
    else:
        raise ValueError(f"Unknown conflict strategy: {strategy}")


# 向后兼容别名（单一实现）
create_default_resolver = create_resolver