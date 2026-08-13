# src/narrative/conflict/__init__.py

from .model import ConflictStrategy, ConflictResolution, StrategyDecision
from .protocol import ConflictResolver
from .provider import StrategyDecisionProvider
from .selectors import RuleSelector  # ✅ 导出
from .strategies import PriorityResolver, BalanceResolver, SynthesisResolver
from .composite import CompositeResolver
from .selector import StrategySelector
from .factory import create_resolver, create_default_resolver
from .default import DefaultConflictResolver

__all__ = [
    "ConflictStrategy",
    "ConflictResolution",
    "StrategyDecision",
    "ConflictResolver",
    "StrategyDecisionProvider",
    "RuleSelector",            # ✅
    "PriorityResolver",
    "BalanceResolver",
    "SynthesisResolver",
    "CompositeResolver",
    "StrategySelector",
    "create_resolver",
    "create_default_resolver",
    "DefaultConflictResolver",
]