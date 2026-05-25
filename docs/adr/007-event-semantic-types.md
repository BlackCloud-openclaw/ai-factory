# ADR-007: Event Semantic Types

## 状态
已接受

## 背景
不同来源的事件（现实获得 vs 梦境获得）对世界状态的“可信度”不同，投影时应有不同处理。

## 决策
- 事件必须携带 semantic 字段，值为：
  - STATE_MUTATION：真实状态改变 → confidence=1.0
  - DIALOGUE / OBSERVATION → 不产生核心谓词
  - DREAM / ILLUSION / FLASHBACK → 投影时 confidence=0.4, priority=flavor
- 投影器根据语义决定如何处理。

## 后果
- 规则可以区分真实与虚幻，避免误判。
- 事件结构更丰富，有利于后续推理。
- 需要确保所有事件都正确标注语义。

## 相关 ADR
- ADR-001: Event is the Source of Truth
- ADR-002: Predicate is Projection Cache