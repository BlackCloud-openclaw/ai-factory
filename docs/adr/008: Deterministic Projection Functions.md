ADR-008: Deterministic Projection Functions

状态：已接受

背景：
投影器必须是确定性的，否则重放结果不可预测，调试困难。

决策：

    DeltaEngine.compute_delta 必须是纯函数，输入仅为 current_active_predicates（字典）和 event。

    禁止在 compute_delta 中使用：datetime.now(), random(), uuid4(), 数据库查询, 外部 API 调用。

    所有时间信息必须从事件的 timestamp 字段获取。

后果：

    相同输入必然产生相同输出，保证可重放性。

    单元测试更容易编写。

    需要注意避免无意中引入非确定性（如字典遍历顺序）。

相关 ADR：ADR-001, ADR-002, ADR-011