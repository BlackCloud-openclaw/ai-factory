# ADR-010: Consistency Budget

## 状态
已接受

## 背景
严格的一致性校验会导致小说失去“意外”和“奇迹”，变得机械。需要允许有限的逻辑漏洞。

## 决策
- 每章分配 max_warnings = 3，max_soft_contradictions = 1。
- Validator 每次返回结果时，若 severity = warning 且通过，消费预算。
- 预算耗尽后，后续 warning 自动升级为 error（必须重试）。
- 预算持久化到 chapter_budget 表，支持断点续写。

## 后果
- 系统在严格一致性与创作自由之间取得平衡。
- 需要维护预算状态。
- 可配置预算值，适应不同风格的小说。

## 相关 ADR
- ADR-004: Rule Engine is Read-Only
- ADR-015