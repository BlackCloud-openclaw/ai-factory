"""
Capability IDs - Framework Capability Vocabulary
Phase 7: 仅包含 ID 常量
Phase 8+: 可扩展为 CapabilitySpec
"""

from src.capabilities.ids import Matchers, Metrics, Repairs, Triggers

__all__ = [
    "Matchers",
    "Metrics",
    "Repairs",
    "Triggers",
]