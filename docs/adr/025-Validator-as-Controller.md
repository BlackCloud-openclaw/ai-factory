
---

## ADR-025: Validator as Controller

```markdown
# ADR-025: Validator as Controller

## 状态
Accepted

## 日期
2026-07-06

## 背景

在 AI Factory v1.0 中，Validator 主要职责是检查"有没有写错"——检查 JSON 格式、检查 must_events 是否出现、检查字数是否足够。它是一个**质量检查器（Quality Checker）**，而非控制器。

v2.0 的研究（Phase 1.5 Semantic Control Spike）发现，真正需要的是**验证 Planner 是否真的控制了 Writer**。这不是"写得好不好"的问题，而是"Writer 是否实现了 Planner 的承诺"的问题。

## 决策

**将 Validator 升级为 Controller，负责验证闭环：**
Planner Promise → Writer → State Change → Validator → Control Score


### 三层 Control

| 层级 | 名称 | 验证内容 | 测量方式 |
|------|------|---------|---------|
| L1 | **Surface Control** | Execution Units 是否被完成 | 关键词匹配 |
| L2 | **Constraint Control** | 硬约束是否被遵守 | 关键词匹配（required/forbidden） |
| L3 | **Outcome Control** | 状态变化是否发生 | 事件匹配（plot_flag/relationship/inventory/realm/location/hp） |

### 评估结果

Phase 1.5 实验表明，三层 Control 可以量化：

| 组 | Surface | Constraint | Outcome |
|---|---------|-----------|---------|
| A | 0.500 | 0.000 | 0.000 |
| B | 0.688 | 0.125 | 0.000 |
| C | 0.447 | 0.105 | 0.000 |
| D | **0.502** | **0.375** | **0.000** |
| E | 0.400 | 0.333 | 0.000 |
| F | 0.546 | 0.250 | 0.000 |
| G | 0.222 | 0.222 | 0.000 |

> 注：Outcome Control 在 Phase 1.5 时为 0，因为 Writer 尚未生成 Observable 事件。Phase 2/3 已修复。

### Validator 职责

1. **验证 Contract 遵循度**：检查 Writer 是否完成了 Contract 中的承诺
2. **计算 Control Scores**：输出 Surface/Constraint/Outcome 分数
3. **触发重试决策**：如果 Control 分数过低，决定是否重试
4. **驱动 Loop 推进**：验证通过后更新 Loop 进度

### 与控制器的集成

```python
# validate_node 中的核心流程
result = await validator.run(state)
if not result["passed"]:
    if should_retry:
        return retry_state
    else:
        return skip_scene

# 验证通过，应用事件
cmd = SceneCompletionCommand(...)
result = await SceneCompletionService.execute(cmd)

# 更新 Loop 进度
if result.chapter_finished:
    await loop_store.update_progress(loop_id, loop_advancement_score)

状态

✅ 已验证

    三层 Control 体系已实现并集成到 _validate_novel_enhanced()

    Validator 成功识别并触发重试

    Loop 进度正常推进（每章 +5%）

    200 章连续测试通过

后果
正面

    Validator 从"质量检查器"升级为"控制器"

    闭环完整：Planner → Contract → Writer → Validator → State Change

    可以量化"控制程度"（Control Scores）

    支持自动重试和降级决策

负面

    Validator 复杂度增加（三层 Control 计算）

    需要 Contract 提供 observables 和 constraints 字段

    Outcome Control 依赖 Writer 正确生成事件

未来扩展

    支持更多 Observable 类型

    支持自定义 Control 规则

    支持 Control 分数的历史趋势分析

参考

    Phase 1.5 实验数据（三层 Control 验证）

    src/agents/validator.py（_validate_contract_units、_validate_contract_constraints、_validate_contract_observables）

    200 章测试日志（Validator 正常执行）