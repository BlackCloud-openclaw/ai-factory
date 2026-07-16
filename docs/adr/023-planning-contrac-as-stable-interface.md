# ADR-023: Planning Contract as Stable Interface

## 状态
Accepted

## 日期
2026-07-06

## 背景

在 AI Factory v1.0 中，Planner 的输出是"场景计划"——一个包含 goal、conflict、must_events 等字段的自由格式字典。Writer 将其视为"建议"，可以选择执行、部分执行或忽略。这导致 Planner 对 Writer 缺乏真正的控制力。

v2.0 的研究（Phase 1 Planning Representation Spike）验证了多种规划表示方式（Summary/Beat/Action/Intent/Constraint），并发现 **Action + Dense** 是最优的规划表示方式（Overall 0.292）。

然而，更大的发现是：**规划内容本身（信息密度）比表示方式更重要。** 更重要的是，Planner 需要一个**稳定的输出格式**——一个可以被 Writer、Validator 和未来 Runtime 共同理解的 DSL。

## 决策

**建立 Planning Contract v1.0 作为 AI Factory 的稳定规划接口。**

Planning Contract 是 Planner 对场景的**承诺（Promise）**，而不是"写作建议"。

### Contract 结构

```yaml
planning_contract:
  version: "1.0"
  scene_id: string

  # 故事意图
  intent:
    goal: string
    conflict: string
    expected_outcome: string

  # 执行计划（唯一的可变层）
  execution:
    kind: string        # action | beat | intent | constraint
    units:
      - id: string
        label: string   # action | beat | intent | constraint
        description: string
        attributes: {}

  # 可观测结果（Planner 对世界的预期）
  observables:
    state_changes:
      - type: plot_flag | relationship | inventory | realm | location | hp
        # 根据 type 不同包含不同字段

  # 硬约束
  constraints:
    - type: required | forbidden | before | after | exclusive | at_least_once
      target: string

  # 领域元数据（不含实验变量）
  metadata:
    chapter: int
    scene_index: int
    arc: optional[string]
设计原则

    Contract 是冻结的 DSL — Planner 可以变，Writer 可以变，Validator 可以变，Runtime 可以变，但 Contract 不变

    不包含 Runtime 概念 — 没有 Event、Projection、Operation、Validator

    不包含实验变量 — representation、density、temperature 属于 ExperimentConfig

    支持版本迁移 — 从第一天就支持 ContractUpcaster

版本迁移
python

class ContractUpcaster:
    @staticmethod
    def upcast(data: dict) -> dict:
        version = data.get("version", "0.9")
        if version == "0.9":
            data = _upcast_v0_9_to_v1_0(data)
        return data

状态

✅ 已验证（Phase 1.5 Semantic Control Spike）

    20+ Contract 示例通过 Schema 验证

    Upcaster 成功迁移旧格式

    Planner 输出 Contract 已集成

    Writer 消费 Contract 已集成

    Validator 验证 Contract 已集成

后果
正面

    Planner、Writer、Validator 之间的接口冻结，允许独立演进

    所有未来 Spike 都建立在同一个稳定接口上

    实验变量与 Contract 分离，避免 Contract 膨胀

    Runtime 可以安全地基于 Contract 构建

负面

    需要为所有新场景编写 Contract 转换逻辑

    旧格式场景计划需要经过 Upcaster 迁移

迁移路径

现有 scene_plan 格式通过 create_contract_from_dict() 自动迁移到 v1.0，无需修改现有代码。
参考

    Phase 1 实验报告：Action + Dense 最优（Overall 0.292）

    src/writing/planning_contract.py

    experiments/phase1/contract_examples/（20+ 示例）