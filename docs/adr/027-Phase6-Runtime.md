好的，我将 Phase 6 的完整架构决策整理为 ADR 027。

---

# ADR 027: Phase 6 Runtime 架构

**状态**: 已采纳（Accepted）  
**日期**: 2026-07-16  
**决策者**: Phase 6 团队  
**影响范围**: Runtime 核心、Workflow 编排、Observability 系统

---

## 背景

在 Phase 6 之前，系统缺乏对 LLM 生成内容的精确控制机制。生成文本的质量依赖于 Prompt Engineering 和事后人工审核，无法在 Runtime 层面进行结构化约束、验证和自动修订。

Benchmark 数据表明：
- 平均合规率仅为 0.161
- 14 个场景中 0 个达到完全合规
- Revision 失败率极高，且无法定位失败原因

核心问题是：**Runtime 缺乏对自然语言输出的结构化控制和可观测性。**

---

## 决策

建立一套 **Compiler-driven 的 IR 控制体系**，包含三个独立系统：

### 1. Computation System（计算系统）

Runtime 是纯计算层，只负责确定性转换。

- **形式**: 无状态 Compiler，输入 IR → 输出 IR
- **位置**: `src/runtime/`
- **组件**:
  - `ObservationCompiler`: Draft → ObservationIR
  - `Validator`: ObservationIR → ComplianceReport
  - `EditCompiler`: Diagnosis → EditPlan
  - `PatchRenderer`: EditPlan → RenderedPatch

### 2. Orchestration System（编排系统）

Workflow 是唯一的流程控制者，负责 LLM 调用、循环、重试和状态管理。

- **形式**: 有状态 Workflow，持有 Session 对象
- **位置**: `src/workflow/`
- **组件**:
  - `RevisionWorkflow`: 修订闭环编排
  - `RevisionSession`: 单一状态对象

### 3. Observation System（观测系统）

被动的事件消费者，不参与业务决策。

- **形式**: Event-driven, 发布-订阅
- **位置**: `src/observability/`
- **组件**:
  - `EventBus`: 事件总线
  - `TraceRecorder`: 执行轨迹记录
  - `MetricsRecorder`: 性能指标记录

### 依赖方向

```
Workflow → Runtime
Workflow → Observability
Runtime ✗ Observability（完全隔离）
```

---

## 核心 IR 体系

| IR | 职责 | 生产者 |
| :--- | :--- | :--- |
| **Decision IR** | 系统“希望”发生什么 | PolicyCompiler |
| **Semantic IR** | 如何表达控制语义 | PolicyCompiler |
| **Observation IR** | 系统实际观察到的文本事实 | ObservationCompiler |
| **Diagnosis IR** | Observation 与 Semantic 的偏差 | Validator |
| **Revision IR** | 如何修复偏差 | EditCompiler |

---

## 九条架构原则

1. **IR Immutability**: 所有 IR 一旦生成，不可变
2. **Observation Monotonicity**: 下游不得重新解析 Draft
3. **Grounded Observation**: 诊断必须建立在 ObservationIR 之上
4. **Reference Stability**: 后续 IR 只引用 ID，不复制内容
5. **Snapshot-based Recompilation**: 每个新 Draft 完整重编译为新 IR
6. **Renderer Boundary**: 自然语言仅存在于 Renderer 与 Engine 之间
7. **Single Compiler Ownership**: 每种 IR 有且仅有一个 Compiler 负责生成
8. **Execution Observability**: Runtime 必须可被完整观测
9. **Workflow Orchestration**: Runtime 只负责计算，不负责控制流程

---

## ExecutionResult 结构

```python
{
    "final_text": str,
    "compliance": float,
    "before_compliance": float,
    "after_compliance": float,
    "compliance_delta": float,
    "stages": [
        {
            "stage": "validation" | "edit_plan" | "patch_render" | "llm" | "revalidation",
            "status": "completed" | "skipped" | "failed",
            "duration_ms": float,
            "payload": dict
        }
    ],
    "artifacts": {
        "validation": ComplianceReport,
        "edit_plan": EditPlan,
        "rendered_prompt": RenderedPatch,
        "llm_output": str,
        "revalidation": ComplianceReport
    }
}
```

---

## 验证结果

| 指标 | Phase 6.3B | Phase 6.3C-2 | 变化 |
| :--- | :--- | :--- | :--- |
| 平均合规率 | 0.161 | 1.000 | +0.839 |
| 改进场景数 | 0/14 | 9/14 | — |
| 完全合规场景数 | 0 | 14/14 | — |
| Runtime Stability Score | — | 1.000 | — |

---

## 与 Phase 7 的边界

Phase 6 完成以下成果：
- ✅ Runtime 纯计算化
- ✅ Workflow 提取
- ✅ Observability 接入
- ✅ Explainability 实现
- ✅ Single Output Contract Principle

Phase 7 的目标：
> 验证同一个 Compiler-driven Runtime，能否无修改地承载新的 Control Surface（Dialogue、Emotion、Pacing 等）。

---

## 相关文件

- `src/runtime/`: 计算系统
- `src/workflow/`: 编排系统
- `src/observability/`: 观测系统
- `src/agents/writer.py`: Writer Agent（调用 Workflow）
- `experiments/phase6/`: 验证脚本和报告

---

## 参考资料

- Phase 6.3A: Runtime 架构设计
- Phase 6.3B: Baseline Benchmark
- Phase 6.3C-1: ObservationCompiler Validation
- Phase 6.3C-2: ExecutionIR Benchmark
- Phase 6.3D: Closed-loop Stability Validation
- Phase 6.4A: Runtime 纯计算化
- Phase 6.4B: Workflow 提取
- Phase 6.5: Execution Facts
- Phase 6.5.1: Explainability
- Revision Executor Validation: LLM Output Contract