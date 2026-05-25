# ADR-012: Snapshot Schema Version

## 状态
已接受

## 背景
WorldState 快照可能因代码变化而格式不兼容，直接加载会导致错误。

## 决策
- world_snapshots 表增加 snapshot_schema_version 字段。
- 加载快照时，若版本低于 CURRENT_SNAPSHOT_VERSION，则丢弃快照，从事件流全量重建。
- 版本号仅当 WorldState 结构发生 breaking change 时递增。

## 后果
- 保证系统升级后仍能正确恢复。
- 避免了因快照格式错误导致的静默数据损坏。
- 需要维护版本常量。

## 相关 ADR
- ADR-001: Event is the Source of Truth
- ADR-011: Event Schema Versioning