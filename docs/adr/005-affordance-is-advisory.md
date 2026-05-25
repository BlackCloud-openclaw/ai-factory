# ADR-005: Affordance is Advisory

## 状态
已接受

## 背景
如果将可供性作为硬约束，会严重限制 LLM 的创造力，导致剧情僵化。

## 决策
- Affordance 仅作为自然语言提示注入到 Planner 的 prompt 中。
- Planner 可以选择遵循或忽略 affordance。
- 禁止在规划阶段强制过滤事件类型。

## 后果
- 保持 LLM 的创作自由度。
- Affordance 仍然可以引导模型向合理方向发展。
- 需要设计好的提示模板，避免过度影响。

## 相关 ADR
- ADR-004: Rule Engine is Read-Only
- ADR-016