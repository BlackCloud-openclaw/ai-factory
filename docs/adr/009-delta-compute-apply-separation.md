# ADR-009: Delta Compute/Apply Separation

## 状态
已接受

## 背景
将计算和应用分离，便于测试、调试、回放和版本迁移。

## 决策
- DeltaEngine 负责计算 PredicateDelta（纯函数）。
- ProjectionStore 负责将 Delta 应用到数据库（处理幂等、冲突、历史）。
- 计算和应用之间通过 PredicateDelta 数据对象通信。

## 后果
- 可以独立测试计算逻辑（不需要数据库）。
- 可以 replay 过去的 Delta 序列，用于调试。
- 投影器版本升级时，可以只更换 DeltaEngine。

## 相关 ADR
- ADR-002: Predicate is Projection Cache
- ADR-008: Deterministic Projection Functions
- ADR-012