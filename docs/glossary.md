# 核心术语表

| 术语 | 定义 |
|------|------|
| **Event** | 不可变历史事实，唯一真相源 |
| **WorldState** | 从事件流重放得到的当前世界状态 |
| **Predicate** | WorldState 的投影缓存，用于快速逻辑校验 |
| **Predicate Identity** | `(subject, relation, normalized_object)` 唯一标识一个谓词 |
| **Predicate Delta** | 一组要激活和要失效的谓词，由投影计算产生 |
| **Projection** | 将事件转换为 Predicate Delta 并应用到缓存的过程 |
| **Replay** | 从事件流全量或增量重建 WorldState / Predicate |
| **Drift** | 缓存投影与全量重放结果不一致 |
| **Affordance** | 当前世界状态下“合理可能发生”的叙事行动，仅作建议 |
| **Consistency Budget** | 每章允许的警告/软矛盾次数 |
| **Core Predicate** | 必须强一致的单值谓词（realm/is_alive/location） |
| **Narrative Predicate** | 允许短期漂移的关系、状态 |
| **Flavor Predicate** | 低重要性软状态（情绪、天气等） |
| **Event Semantic** | 事件语义分类（STATE_MUTATION, DIALOGUE, DREAM, ILLUSION 等） |
| **Event Upcaster** | 将旧版本事件升级到最新 schema 的转换器 |
| **Dead Letter** | 投影器无法处理的毒化事件，进入死信队列 |