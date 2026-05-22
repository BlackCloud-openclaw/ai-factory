ADR-015: Chapter Consistency Budget Persistence

状态：已接受

背景：
ConsistencyBudget 如果仅保存在内存中，服务重启或章节续写时预算会丢失，导致行为不一致。

决策：

    创建 chapter_budget 表，以 (novel_id, volume_num, chapter_num) 为主键。

    Validator 每次消费预算时，同步更新数据库。

    续写时从数据库加载预算，保证连续性。

后果：

    预算语义持久化，符合确定性原则。

    需要额外的数据库写操作。

    可以支持多实例协同（通过数据库共享状态）。

相关 ADR：ADR-004, ADR-010