# ADR-016: Affordance Cooldown

## 状态
已接受

## 背景
如果 affordance 一直推荐同一能力（如“可以御剑飞行”），LLM 会过度使用，导致叙事重复。

## 决策
- 规则配置中可指定 cooldown: N（冷却章节数）。
- 每次 affordance 被实际使用（出现在 prompt 中）后，记录 (novel_id, affordance_id, last_used_chapter)。
- 后续计算该 affordance 分数时，若距离上次使用不足 N 章，得分乘以 0.2（大幅降权）。

## 后果
- 增加叙事多样性，避免能力滥用。
- 需要维护 affordance_usage 表。
- 可配置冷却值，适应不同故事节奏。

## 相关 ADR
- ADR-005: Affordance is Advisory
- ADR-007: Event Semantic Types