# ADR-019: WorldState Schema V2

## 状态
已接受

## 背景
WorldState 数据结构经历了多次演进，需要明确的 schema 版本标记以支持：
- 向后兼容旧快照
- 未来 schema 升级
- 确定性迁移

## 决策
- 使用 `schema_version: int` 作为数据结构版本（不与产品版本混淆）
- `schema_version = 2` 表示 characters 使用 ID-key
- `schema_version = 1` 表示 characters 使用 name-key（旧格式）
- 加载时自动检测并迁移旧版本
- 新写入的 snapshot 使用 `schema_version = 2`

## 后果
- ✅ 明确的迁移边界
- ✅ 未来 schema 升级可预测
- ✅ 与 ADR-012 的 snapshot schema 版本一致
- ⚠️ 需维护 `migrate_v1_to_v2()` 迁移函数

## 相关 ADR
- ADR-001: Event is the Source of Truth
- ADR-011: Event Schema Versioning
- ADR-012: Snapshot Schema Version
- ADR-017: Schema Compatibility Policy
- ADR-018: Character Name → Character ID Migration