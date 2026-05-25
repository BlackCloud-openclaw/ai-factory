# ADR-006: Core Predicate Singleton Policy

## 状态
已接受

## 背景
某些谓词（如 realm, is_alive, location）在逻辑上应保证同一时间只有一个活跃值。

## 决策
- 定义 SINGLETON_RELATIONS = {"realm", "is_alive", "location"}。
- 激活新值时，ProjectionStore 自动将同一 (subject, relation) 的当前活跃记录标记为失效（is_active=false, valid_to_event_id=当前事件ID）。
- 其他关系（如 has_item）允许多条同时活跃。

## 后果
- 保证核心语义一致性。
- 避免了同时存在“金丹期”和“炼气期”的矛盾。
- 实现复杂度略有增加，但收益显著。

## 相关 ADR
- ADR-001: Event is the Source of Truth
- ADR-002: Predicate is Projection Cache
- ADR-013