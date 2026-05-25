# ADR-004: Rule Engine is Read-Only

## 状态
已接受

## 背景
规则引擎如果能够修改状态，会绕过事件溯源，破坏可重放性。

## 决策
- 规则引擎仅用于校验（Validator）和推导（Affordance）。
- 规则不产生事件、不直接写 WorldState 或 predicates 表。
- 规则匹配结果：允许/拒绝/警告，以及缺失的前提条件列表。

## 后果
- 世界状态的变化必须通过事件，保持单一真相源。
- 规则引擎可以安全地并行执行（只读）。
- 规则的副作用仅限于日志和指标。

## 相关 ADR
- ADR-001: Event is the Source of Truth
- ADR-013