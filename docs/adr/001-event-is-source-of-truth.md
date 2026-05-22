# ADR-001: Event is the Source of Truth

## 状态
已接受

## 背景
在叙事生成系统中，需要保证状态的可重放、可审计和时间旅行能力。

## 决策
- 所有状态变更必须通过 `NarrativeEvent` 表达。
- `WorldState` 和 `Predicate` 均为从事件流派生的视图，可随时重建。
- 禁止直接修改 `WorldState` 或 `Predicate` 表而不经过事件。

## 后果
- 必须维护完整的事件日志。
- 投影层必须支持全量重建和增量更新。
- 快照仅用于性能优化，不是真相源。

## 相关 ADR
- ADR-002: Predicate is Projection Cache
- ADR-008: Deterministic Projection Functions