ADR-013: Projection Invariant Assertions

状态：已接受

背景：
投影错误（如重复激活、单值关系冲突）可能在运行中悄然发生，需要运行时检查。

决策：

    InvariantChecker 在每次 apply_delta 后检查受影响的实体。

    核心不变量：

        单值关系最多一条活跃记录。

        置信度在 0-1 之间。

        valid_from_event_id <= valid_to_event_id（若不为空）。

    违反不变量时：日志 + 递增指标 + 可选触发告警（不阻断正常流程）。

后果：

    及早发现投影逻辑 bug。

    便于调试和修复。

    轻微性能开销（仅检查受影响实体）。

相关 ADR：ADR-002, ADR-006, ADR-008