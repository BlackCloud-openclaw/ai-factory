# src/narrative/conflict/default.py

from .strategies.priority import PriorityResolver

# 向后兼容别名（单一实现，无独立逻辑）
DefaultConflictResolver = PriorityResolver