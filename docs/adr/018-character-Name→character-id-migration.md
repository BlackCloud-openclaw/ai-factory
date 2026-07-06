# ADR-018: Character Name → Character ID Migration

## 状态
已接受

## 背景
WorldState.characters 的键使用角色的 `name`（可变的显示名称），导致：
- 改名后需迁移所有历史快照和事件
- 无法通过稳定标识引用角色
- Domain Plugin 无法适配不同题材（不同题材的主角名不同）

## 决策
- WorldState.characters 使用 `character_id` 作为唯一主键
- `name` 保留为显示名称，仅用于日志和最终输出
- 业务逻辑使用 ID 操作，使用 `get_character()` 兼容名称查找
- 快照保存时自动将 name-key 转换为 ID-key

## 后果
- ✅ 改名不影响状态，无需迁移历史数据
- ✅ 支持 Domain Plugin 多题材适配
- ✅ 为 Kernel Entity 系统铺平道路
- ⚠️ 需要快照迁移兼容层（Phase 4C 已实现）
- ⚠️ 禁止新增代码使用 `world.characters[name]`

## 相关 ADR
- ADR-001: Event is the Source of Truth
- ADR-002: Predicate is Projection Cache
- ADR-011: Event Schema Versioning
- ADR-012: Snapshot Schema Version
- ADR-019: WorldState Schema V2
- ADR-020: Config-Driven Identity