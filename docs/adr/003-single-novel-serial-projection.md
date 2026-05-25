# ADR-003: Single-Novel Serial Projection

## 状态
已接受

## 背景
并行处理同一小说的投影会导致乱序，破坏 event_id 顺序依赖，引发状态漂移。

## 决策
- 同一 novel_id 的投影任务必须串行执行。
- 实现方式：PostgreSQL 行锁（SELECT ... FOR UPDATE）或单消费者分区队列。
- 投影 worker 每次获取 last_projected_event_id 后，按 event_id 严格递增顺序处理。

## 后果
- 投影吞吐量受限于单 novel 串行处理。
- 避免了因果错乱和重复应用。
- 需要监控投影延迟。

## 相关 ADR
- ADR-001: Event is the Source of Truth
- ADR-002: Predicate is Projection Cache
- ADR-008: Deterministic Projection Functions