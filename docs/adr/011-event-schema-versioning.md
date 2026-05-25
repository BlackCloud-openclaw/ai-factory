# ADR-011: Event Schema Versioning

## 状态
已接受

## 背景
事件结构会随着系统演化而变化，需要支持向后兼容的历史事件重放。

## 决策
- narrative_events 表增加 event_schema_version 字段（默认 1）。
- 提供 EventUpcaster 类，按版本链将事件升级到最新 schema。
- 投影器和 WorldState 重放始终使用最新 schema。

## 后果
- 旧事件可以无缝使用新逻辑。
- 需要谨慎设计升级函数，保证可逆（至少可向前）。
- 增加了少量运行时开销。

## 相关 ADR
- ADR-001: Event is the Source of Truth
- ADR-008: Deterministic Projection Functions