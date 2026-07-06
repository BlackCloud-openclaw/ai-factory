# ADR-020: Config-Driven Identity

## 状态
已接受

## 背景
角色身份散落在代码中（验证器中的硬编码、Prompt 示例、状态投影中的名字），导致：
- 更换题材需要修改多处代码
- 角色改名需要全局搜索替换
- 无法通过配置切换题材

## 决策
- 所有角色身份由 YAML 配置驱动（`config/{domain}/character.yaml`）
- 使用 `id` 作为系统标识，`name` 作为显示名称
- 通过 `tags` 进行角色分类（如 `main` 标签标记主角）
- 提供 `src/domain/identity.py` 统一访问接口
- 禁止在业务代码中直接使用角色名字符串
- 配置加载支持缓存清理（`reload_character_config()`），为热加载预留

## 后果
- ✅ 题材切换仅需修改 YAML 配置
- ✅ 角色改名无需改代码
- ✅ 符合 Kernel 化目标，与 Domain Plugin 兼容
- ⚠️ 需要 Contract Test 验证配置完整性
- ⚠️ 需维护配置变更时的兼容性

## 相关 ADR
- ADR-018: Character Name → Character ID Migration
- ADR-019: WorldState Schema V2
- ADR-017: Schema Compatibility Policy