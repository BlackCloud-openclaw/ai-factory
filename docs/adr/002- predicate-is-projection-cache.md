# ADR-002: Predicate is Projection Cache

## 状态
已接受

## 背景
事件溯源系统中，直接查询事件流或 WorldState 进行频繁的逻辑校验效率低下。需要一个轻量级的、面向规则的视图层。

## 决策：
- Predicate 是从 WorldState 投影得到的只读缓存，不是真相源。
- predicates 表存储投影结果，不设唯一约束，允许多版本共存。
- 每次事件写入后，通过 PredicateDelta 增量更新缓存。
- 缓存可随时从事件流全量重建。

## 后果
- 投影层必须支持幂等、顺序保证、历史版本。
- 缓存与 WorldState 可能短暂不一致（最终一致性）。
- 投影逻辑的变更需要递增 projection_version。

## 相关 ADR
ADR-001, ADR-003, ADR-008, ADR-013